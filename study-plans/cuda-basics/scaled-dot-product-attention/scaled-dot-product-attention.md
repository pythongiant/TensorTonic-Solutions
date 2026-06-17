# <span style="font-size: 20px;">Scaled Dot-Product Attention</span>

<span style="font-size: 14px;">Attention computes $\text{out} = \text{softmax}(QK^\top / \sqrt{D})\,V$, mixing every query against every key and blending the values by the resulting weights. It is the capstone of the CUDA track because it is not one pattern but a **pipeline** of three: a matmul, a row-parallel reduction, and a second matmul. The systems story is the $N \times N$ scores matrix sitting in the middle - the buffer that dominates memory traffic and motivates fusion. Composing the GEMM and softmax kernels from earlier problems exposes exactly why materializing that matrix is the bottleneck and why Flash-Attention avoids it.</span>

---

## The Operation, $\text{out} = \text{softmax}(QK^\top / \sqrt{D})\,V$

<span style="font-size: 14px;">With $Q$, $K$, $V$ each of shape $(N, D)$, the computation factors into three stages:</span>

$$
S = \frac{QK^\top}{\sqrt{D}}, \qquad P = \text{softmax}(S), \qquad \text{out} = P V
$$

<span style="font-size: 14px;">$S$ and $P$ are $N \times N$; the softmax is taken over each row of $S$. All buffers are contiguous row-major, so element $S[i,j]$ lives at offset $i \cdot N + j$. The $1/\sqrt{D}$ scale keeps the dot products from growing with $D$ and saturating the softmax. The output is back to shape $(N, D)$.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy: A Three-Stage Pipeline</span>

<span style="font-size: 14px;">The clean implementation is three separate kernel launches, each a pattern already seen:</span>

<span style="font-size: 14px;">1. **Scores GEMM**: a tiled matrix multiply computes $QK^\top$ and multiplies by $1/\sqrt{D}$, writing the $N \times N$ scores buffer $S$ to global memory. Each output element is a length-$D$ dot product of a $Q$ row and a $K$ row, so $K$ is read transposed, which the tiling stages into `__shared__` memory to keep loads coalesced.</span>

<span style="font-size: 14px;">2. **Row softmax**: a row-parallel kernel, one block per row of $S$, computes the stable softmax in place - a max reduction, then a sum-of-exponentials reduction, both in `__shared__` memory and separated by `__syncthreads()`, then a normalize. This is the cross-entropy reduction pattern applied to $N$ rows.</span>

<span style="font-size: 14px;">3. **Output GEMM**: a second tiled matmul multiplies the normalized weights $P$ by $V$, producing the $(N, D)$ output. Each output row is a weighted sum of all $V$ rows, again tiled through shared memory.</span>

<span style="font-size: 14px;">The two GEMMs map output tiles to thread blocks; the softmax maps rows to blocks. Each stage is correct and reusable on its own, which is the pedagogical point: attention is the earlier patterns composed.</span>

---

## <span style="font-size: 16px;">Kernel Boundaries Are Synchronization Points</span>

<span style="font-size: 14px;">The three kernels run in sequence on the same stream, and a kernel launch is an implicit barrier: stage 2 cannot start until every block of stage 1 has finished writing $S$, because the softmax of a row needs the whole row. The GPU has no cheap global barrier within a single launch, so the launch boundary is how the pipeline synchronizes. Splitting attention into three kernels is therefore not just modular - it is the mechanism that guarantees all of $S$ exists before the softmax reads it, and all of $P$ exists before the output GEMM reads it.</span>

<span style="font-size: 14px;">That structure also defines the **scratch-buffer lifetime**. The $N \times N$ buffer $S$ is allocated before stage 1, written by stage 1, read and overwritten by stage 2, read by stage 3, and only then freed. It lives across all three launches and is the single largest allocation in the algorithm.</span>

<span style="font-size: 14px;">This is also where the memory cost becomes a hard wall rather than a slowdown. The scratch buffer grows as $N^2$, so doubling the sequence length quadruples the allocation; at long context lengths the $N \times N$ scores simply do not fit in device memory, which is the practical reason the unfused pipeline cannot scale. The launch-boundary synchronization that makes the three-kernel version simple is exactly what forces the full matrix to exist in global memory between stages, because one kernel cannot hand a partial result to the next without writing it to DRAM.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">Inside each GEMM the access pattern is the classic tiling story: a naive matmul rereads a full row of $Q$ and column of $K$ from global memory for every output element, an $O(N)$ reuse factor wasted; tiling stages a `TILE x TILE` block of each operand into `__shared__` memory so each global load is reused `TILE` times, and pads the tile (`[TILE][TILE+1]`) to dodge **bank conflicts**. The softmax stage reads and writes each row of $S$ from global memory and reduces in shared memory.</span>

<span style="font-size: 14px;">The dominant traffic, though, is $S$ itself. Stage 1 writes $N^2$ scores to DRAM, stage 2 reads and rewrites them, stage 3 reads them again - roughly $4 N^2$ elements of global traffic just to shuttle the intermediate, versus the $3ND$ elements of actual input. For the long sequences where attention is used, $N \gg D$, so the $N^2$ scratch traffic dwarfs the $ND$ input traffic and sets the runtime.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">The two GEMMs are individually compute-bound once tiled: their arithmetic intensity grows with tile size until each loaded byte feeds many fused multiply-adds, lifting them above the **roofline** ridge point. But the pipeline as a whole is dragged back toward memory-bound by the $N \times N$ scratch. Counting the round trips of $S$:</span>

$$
\text{scratch traffic} \approx 4 N^2 \text{ elements} \gg 3 N D \text{ input elements}
$$

<span style="font-size: 14px;">So while the multiplies are plentiful, a large fraction of wall-clock time is spent writing and rereading an intermediate that exists only because the stages are separate. The compute-bound GEMMs are bottlenecked by a memory-bound handoff. That mismatch - useful compute throttled by avoidable traffic - is precisely the gap that fusion closes.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized: The Case for Fusion</span>

<span style="font-size: 14px;">The three-kernel version is the naive baseline: correct, readable, and dominated by $N^2$ scratch traffic and the $N^2$ memory footprint, which becomes the limiting resource for long sequences. The optimized form is **Flash-Attention-style fusion**, which never materializes the full scores matrix in global memory.</span>

<span style="font-size: 14px;">The idea is to keep a tile of scores tile-local in `__shared__` memory and fuse all three stages into one kernel that walks the keys and values in blocks. For each block of keys, the kernel computes a tile of $QK^\top$ in shared memory, updates a **running softmax** - a running maximum and a running normalizer carried in registers - and accumulates the partial $PV$ output on the fly, rescaling the accumulator as the running max shifts. Because no row of $S$ is ever fully present, the full matrix is never written to DRAM.</span>

<span style="font-size: 14px;">The payoff is quantified directly: global scratch traffic drops from $O(N^2)$ to $O(N D)$, and the $N \times N$ memory footprint disappears, replaced by tile-sized shared-memory buffers. The cost is recomputing softmax statistics incrementally rather than once - more arithmetic - but since the GEMMs are compute-bound and the bottleneck was the scratch traffic, trading cheap recompute for eliminated DRAM round trips is a large net win and is what makes long-context attention tractable.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 2$, $D = 2$, with $Q = K = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}$ and $V = \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}$, so $\sqrt{D} = \sqrt{2} \approx 1.414$.</span>

* <span style="font-size: 14px;">**Stage 1**: $QK^\top = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}$, scaled by $1/\sqrt{2}$ gives $S \approx \begin{bmatrix}0.707 & 0 \\ 0 & 0.707\end{bmatrix}$, written to the scratch buffer.</span>
* <span style="font-size: 14px;">**Stage 2**: row-softmax of $[0.707, 0]$ is about $[0.670, 0.330]$, and by symmetry row 1 is $[0.330, 0.670]$.</span>
* <span style="font-size: 14px;">**Stage 3**: $P V$ gives row 0 $\approx 0.670\cdot[1,2] + 0.330\cdot[3,4] \approx [1.66, 2.66]$, and row 1 $\approx [2.34, 3.34]$.</span>

<span style="font-size: 14px;">The fused kernel produces the identical output without ever storing the $2 \times 2$ matrix $S$ in global memory: it forms each scores tile in shared memory, folds it into the running softmax, and accumulates into the output. For $N = 2$ the saving is trivial, but the $N^2$ buffer it removes is what dominates at sequence lengths in the thousands.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Materializing the full $N \times N$ scores.** The unfused pipeline's $O(N^2)$ scratch traffic and footprint are the real bottleneck for long sequences; fusion that keeps scores tile-local in shared memory is the fix.</span>
* <span style="font-size: 14px;">**Skipping the max-subtract in the softmax.** Exponentiating unscaled or unshifted scores overflows `expf`; the $1/\sqrt{D}$ scale and the row max are both stability requirements.</span>
* <span style="font-size: 14px;">**Reading $S$ before stage 1 finishes.** The softmax of a row needs the whole row, so it depends on the launch boundary as a barrier; merging stages without preserving that ordering reads partial scores.</span>
* <span style="font-size: 14px;">**Bank conflicts in the GEMM tiles.** Power-of-two tile strides serialize shared-memory access; pad the tile (`[TILE][TILE+1]`) so the matmuls stay conflict-free.</span>

---