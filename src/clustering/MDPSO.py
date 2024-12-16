# src/clustering/MDPSO.py
import torch
from tqdm import tqdm
from params import DEVICE, MIN_DIM, MAX_DIM, NUM_PARTICLES, MAX_ITERATION, W_0, W_N, C1, C2
from clustering.fitness import fitness_function, construct_clustering
from src.clustering.Particle import Particle


class MDPSO:
    def __init__(self, image, bounds, weights, device=DEVICE):
        """
        Initialize the MDPSO algorithm.
        :param image: The MRI image (H x W x 4 modalities).
        :param bounds: The bounds of the optimization process.
        :param weights: The weights for the weighted sum in the fitness function.
        :param device: The CUDA device to use.
        """
        self.image = image
        self.bounds = bounds
        self.weights = weights
        self.device = device
        self.streams = [torch.cuda.Stream() for _ in range(NUM_PARTICLES)]  # CUDA streams
        self.dimensions = list(range(MIN_DIM, MAX_DIM + 1))
        self.global_bests = {dim: {'position': None, 'fitness': float('inf')} for dim in self.dimensions}

        # Prepare for fitness calculations
        H, W, _ = image.shape
        x_coords = torch.linspace(0, 1, W).repeat(H, 1).to(DEVICE)
        y_coords = torch.linspace(0, 1, H).repeat(W, 1).T.to(DEVICE)
        full_image = torch.cat((
            x_coords.unsqueeze(-1),
            y_coords.unsqueeze(-1),
            image
        ), dim=-1)  # (H, W, 6)
        # Cluster only the points that are non-zero in all modalities (assuming zeros are background)
        self.flat_image = full_image.view(-1, 6)[(full_image.view(-1, 6)[:, 2:] != 0).any(dim=1)]

        # Initialize particles
        self.particles = self.initialize_particles_streams()

    def initialize_particles_streams(self):
        """Initialize particles using CUDA streams."""
        streams = [torch.cuda.Stream() for _ in range(NUM_PARTICLES)]  # Create CUDA streams
        particles = [None] * NUM_PARTICLES  # Space for initialized particles

        for i in range(NUM_PARTICLES):
            with torch.cuda.stream(streams[i]):  # Each stream initializes one particle
                particles[i] = Particle(
                    self.dimensions, self.bounds, self.device, self.image, fitness_function, self.weights, self.flat_image
                )

        torch.cuda.synchronize()  # Wait for all streams to complete
        return particles

    def optimize(self):
        """Multi-Dimensional Particle Swarm Optimization."""
        dimensions = list(range(MIN_DIM, MAX_DIM + 1))
        global_bests = {dim: {'position': None, 'fitness': float('inf')} for dim in dimensions}

        # Initialize global bests for each dimension
        for dim in dimensions:
            best_particle = min(self.particles, key=lambda p: p.local_bests[dim]['fitness'])
            global_bests[dim]['position'] = best_particle.local_bests[dim]['position'].clone()
            global_bests[dim]['fitness'] = best_particle.local_bests[dim]['fitness']

        global_best_dim = min(global_bests, key=lambda d: global_bests[d]['fitness'])

        fitness_history = []  # Store the global best fitness history

        # Preinitialize CUDA streams
        streams = [torch.cuda.Stream() for _ in range(NUM_PARTICLES)]
        fitness_results = [None] * NUM_PARTICLES

        # Algorithm iterations
        for iteration in tqdm(range(MAX_ITERATION)):
            inertia = W_0 - iteration * (W_0 - W_N) / MAX_ITERATION

            for i, particle in enumerate(self.particles):
                with torch.cuda.stream(streams[i]):  # Use streams
                    fitness_results[i], _ = fitness_function(particle.positions[particle.dim], self.flat_image, self.weights)

            torch.cuda.synchronize()  # Wait for all fitness calculations to complete

            # Process fitness results
            for i, particle in enumerate(self.particles):
                current_fitness = fitness_results[i]
                if current_fitness < particle.local_bests[particle.dim]['fitness']:
                    particle.local_bests[particle.dim]['fitness'] = current_fitness
                    particle.local_bests[particle.dim]['position'] = particle.positions[particle.dim].clone()

                if current_fitness < global_bests[particle.dim]['fitness']:
                    global_bests[particle.dim]['fitness'] = current_fitness
                    global_bests[particle.dim]['position'] = particle.positions[particle.dim].clone()

            global_best_dim = min(global_bests, key=lambda d: global_bests[d]['fitness'])
            fitness_history.append(global_bests[global_best_dim]['fitness'])

            # Update positions and velocities in parallel
            for i, particle in enumerate(self.particles):
                with torch.cuda.stream(streams[i]):  # Reuse streams
                    particle.update_velocity(global_bests[particle.dim]['position'], inertia, C1, C2)
                    particle.update_centroid()
                    particle.update_dimensional_velocity(global_best_dim, inertia, C1, C2)
                    particle.move_to_new_dimension()

            torch.cuda.synchronize()  # Wait for the iteration to complete

        best_dim = min(global_bests, key=lambda d: global_bests[d]['fitness'])

        _, clust_indices = fitness_function(global_bests[best_dim]['position'], self.flat_image, self.weights)
        clust = construct_clustering(self.flat_image, (240, 240), clust_indices, self.device)

        return clust
