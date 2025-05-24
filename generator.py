import torch

def init_pos(size=None):
    if size == None:
        return torch.randn((1,3))
    else:
        return torch.vstack((torch.tensor([0,0,0]), torch.randn((size - 1, 3))))

def initialize_on_sphere(N, radius=2):
    theta = torch.acos(2 * torch.rand(N - 1) - 1)  
    phi = torch.rand(N - 1) * 2 * torch.pi  

    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)

    positions = torch.stack([x, y, z], dim=1)
    return torch.vstack((torch.tensor([0,0,0]), positions))

def fibonacci_sphere(N, radius=1.2):
    points = []
    phi = torch.tensor(torch.pi * (3.0 - 5.0 ** 0.5))

    for i in range(N):
        y = 1 - (i / float(N - 1)) * 2 
        radius_xy = (1 - y * y) ** 0.5

        x = radius_xy * torch.cos(i * phi)
        z = radius_xy * torch.sin(i * phi)

        points.append(torch.tensor([x, y, z]) * radius)

    return torch.stack(points)

def uniform_sphere(N, radius=1.0):
    points = []
    phi = torch.acos(torch.tensor(1.0) - 2 * (torch.arange(0, N, dtype=torch.float32) + 0.5) / N)  # Polar angles
    phi_angle = torch.tensor(torch.pi * (3.0 - 5.0 ** 0.5)) * torch.arange(0, N, dtype=torch.float32)  # Azimuthal angles
    
    # Convert spherical coordinates to Cartesian coordinates
    x = radius * torch.sin(phi) * torch.cos(phi_angle)
    y = radius * torch.sin(phi) * torch.sin(phi_angle)
    z = radius * torch.cos(phi)
    
    points = torch.stack((x, y, z), dim=1)
    
    return points