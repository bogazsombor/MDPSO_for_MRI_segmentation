# src/clustering/Particle.py
import torch
from params import DIM_STEP, VELOCITY_LIMIT, DIM_VELOCITY_LIMIT, MIN_DIM, MAX_DIM

class Particle:
    def __init__(self, all_dims, bounds, device, image, fitness_function, weights, flat_image):
        self.bounds = bounds
        self.device = device
        self.all_dims = all_dims
        self.image = image
        self.dim_velocity = torch.tensor(0.0, device=device)
        self.weights = weights
        self.flat_image = flat_image

        H, W, _ = image.shape

        self.positions = {
            dim: torch.rand(dim * DIM_STEP * 6, device=device) * (bounds[1] - bounds[0]) + bounds[0]
            for dim in all_dims
        }
        self.velocities = {
            dim: torch.zeros(dim * DIM_STEP * 6, device=device)
            for dim in all_dims
        }
        self.local_bests = {
            dim: {
                'position': self.positions[dim].clone(),
                'fitness': float('inf')
            }
            for dim in all_dims
        }

        for dim in all_dims:
            num_centroids = dim * DIM_STEP
            for i in range(num_centroids):
                x, y = (self.positions[dim][i*6:i*6+2]*torch.tensor([W,H], device=device)).long()
                x = torch.clamp(x,0,W-1)
                y = torch.clamp(y,0,H-1)
                self.positions[dim][i*6+2:i*6+6] = image[y,x,:]

        for dim in all_dims:
            self.local_bests[dim]['fitness'], _ = fitness_function(self.positions[dim], self.flat_image, self.weights)

        self.best_dim = min(self.local_bests, key=lambda d: self.local_bests[d]['fitness'])
        self.best_fitness = self.local_bests[self.best_dim]['fitness']

        random_index = torch.randint(0, len(all_dims), (1,)).item()
        self.dim = all_dims[random_index]

    def update_velocity(self, global_best_position, inertia, cognitive, social):
        num_centroids = self.dim * DIM_STEP * 6
        r1 = torch.rand(num_centroids, device=self.device)
        r2 = torch.rand(num_centroids, device=self.device)
        cognitive_velocity = cognitive * r1 * (self.local_bests[self.dim]['position'] - self.positions[self.dim])
        social_velocity = social * r2 * (global_best_position - self.positions[self.dim])
        self.velocities[self.dim] = inertia * self.velocities[self.dim] + cognitive_velocity + social_velocity
        self.velocities[self.dim] = torch.clamp(self.velocities[self.dim], -VELOCITY_LIMIT, VELOCITY_LIMIT)

    def update_centroid(self):
        H, W, _ = self.image.shape
        num_centroids = self.dim * DIM_STEP
        self.positions[self.dim] += self.velocities[self.dim]
        for i in range(num_centroids):
            self.positions[self.dim][i*6:i*6+2] = torch.clamp(self.positions[self.dim][i*6:i*6+2], 0, 1)
            x, y = (self.positions[self.dim][i*6:i*6+2]*torch.tensor([W-1,H-1], device=self.device)).long()
            self.positions[self.dim][i*6+2:i*6+6] = self.image[y,x,:]

    def update_dimensional_velocity(self, global_best_dim, inertia, cognitive, social):
        r1 = torch.rand(1, device=self.device)
        r2 = torch.rand(1, device=self.device)
        cognitive_velocity = cognitive * r1 * (self.best_dim - self.dim)
        social_velocity = social * r2 * (global_best_dim - self.dim)
        self.dim_velocity = inertia * self.dim_velocity + cognitive_velocity + social_velocity
        self.dim_velocity = torch.clamp(self.dim_velocity, -DIM_VELOCITY_LIMIT, DIM_VELOCITY_LIMIT)

    def move_to_new_dimension(self):
        new_dim = int(self.dim + round(self.dim_velocity.item()))
        if MIN_DIM <= new_dim <= MAX_DIM:
            self.dim = new_dim
