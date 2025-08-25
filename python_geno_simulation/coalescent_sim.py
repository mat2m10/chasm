import numpy as np
import msprime

def simulate_coalescent(sample_size: int, theta: float, sites: int, seed: int | None = None):
    """
    Simulate a panmictic coalescent with neutral mutations.
    Returns haplotypes as a NumPy array (sample_size x sites).
    """
    ts = msprime.sim_ancestry(
        samples=sample_size,
        sequence_length=sites,
        recombination_rate=0,
        random_seed=seed
    )
    ts = msprime.sim_mutations(ts, rate=theta / (4 * sample_size), random_seed=seed)
    return ts.genotype_matrix().T  # rows: samples, cols: sites


def simulate_seqs_ls(sample_size: int, theta: float, rho: float, seq_length: int, seed: int | None = None):
    """
    Simulate coalescent with recombination across a sequence of given length.
    """
    ts = msprime.sim_ancestry(
        samples=sample_size,
        sequence_length=seq_length,
        recombination_rate=rho,
        random_seed=seed
    )
    ts = msprime.sim_mutations(ts, rate=theta / (4 * sample_size), random_seed=seed)
    return ts.genotype_matrix().T


def simulate_bb(n: int, p: float, alpha: float, beta: float, seed: int | None = None):
    """
    Simulate allele counts under a Beta-Binomial model.
    """
    rng = np.random.default_rng(seed)
    q = rng.beta(alpha, beta)
    return rng.binomial(n, p * q)


def simulate_coalescent_split(sample_sizes: tuple[int, int], theta: float, M: float, split_time: float, seq_length: int, seed: int | None = None):
    """
    Two-deme coalescent with migration until split_time, then merged.
    """
    demography = msprime.Demography()
    demography.add_population(name="pop0", initial_size=1000)
    demography.add_population(name="pop1", initial_size=1000)
    demography.set_migration_rate("pop0", "pop1", M)
    demography.set_migration_rate("pop1", "pop0", M)
    demography.add_population_split(time=split_time, derived=["pop0", "pop1"], ancestral="ancestral")

    ts = msprime.sim_ancestry(
        samples={"pop0": sample_sizes[0], "pop1": sample_sizes[1]},
        sequence_length=seq_length,
        demography=demography,
        random_seed=seed
    )
    ts = msprime.sim_mutations(ts, rate=theta / (4 * sum(sample_sizes)), random_seed=seed)
    return ts.genotype_matrix().T
