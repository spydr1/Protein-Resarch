# Definitions

* Group: SO3
* Equivariance:
  * $f:X-Y$ is equivariant to $T_g, Sg$ if for all $\forall g,x f(T_g(x)) = S_g(f(x))$

* For our transformations we look at representations
* representation: $p:G-GL(N)$ where GL(N) is NxN matrix of linear transformations
* representations p,q are equivalent if $p = QqQ^{-1}$
* representation p is reducible if $p = Q p_1 \oplus p_2 Q^{-1}$
* representation p is irreducible if it is not possible to reduce
* **?? Note that each irrep acts on a separate subspce, mapping vectors from that space back into it**
  * This is due to the zero fill in the diagonal structure of the sum
* S is invariant subspace under irred. representation if $p(G)(S) \subset S$

# Representation Theory of SO3
* linear representations of compact group can be directly decomposed into sum of irreducibles st.
* $p = Q \underset{j}{\oplus}(D_j) Q^{-1}$ where D_j are 2j+1x2j+1 complex matrices
* type-j vectors  where $Q=I$ 
  * **?? type-0 invariant under rotations?, type-1 rotation under 3d rotation mat?**

* Orthogonal space where representation acts on is the space of spherical harmonics? (What is this space of $\mathbb{C}^{2j+1}$)

# Spherical Harmonics
* Function $Y_j:S2-\mathbb{C}^{2j+1}$
* Basis of all integratible functions on S2
* $Y_j(R_g x ) = D_j Y_j(x)$

# Need to understand (measure theory review)
* Hilbert space
* sq integratible functions
* role of inner product
