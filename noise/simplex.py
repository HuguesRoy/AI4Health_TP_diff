import torch
import math

def rand_perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[-2], res[1] / shape[-1])
    d = (shape[-2] // res[0], shape[-1] // res[1])
    
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
    if len(shape)>2:

        angles = 2*math.pi*torch.rand(shape[0],res[0]+1, res[1]+1)
    else:
        angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)

    def dot(grad, shift):
        first = torch.stack((grid[:shape[-2],:shape[-1],0] + shift[0], grid[:shape[-2],:shape[-1], 1] + shift[1]  ), dim = -1)
        second = grad[...,:shape[-2], :shape[-1]]
        # print("first: ",first.shape)
        # print("second :",second.shape)
        return (first * second).sum(dim = -1)
    #dot = lambda grad, shift: (torch.stack((grid[:shape[-2],:shape[-1],0] + shift[0], grid[:shape[-2],:shape[-1], 1] + shift[1]  ), dim = -1) * grad[...,:shape[-2], :shape[-1]]).sum(dim = -1)
    
    if len(shape)>2:
        
        tile_grads = lambda slice1, slice2: gradients[...,slice1[0]:slice1[1],slice2[0]:slice2[1],:].repeat_interleave(d[0], -3).repeat_interleave(d[1], -2)
        # print("tiles_grad: ",tile_grads([0, -1], [0, -1]).shape)
        n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
        t = fade(grid[:shape[-2], :shape[-1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
    else:
        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1],slice2[0]:slice2[1] ].repeat_interleave(d[0], -3).repeat_interleave(d[1], -2)
        n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for i in range(octaves):
        # print("i = ,",i)
        # print((frequency*res[0], frequency*res[1]))
        noise += amplitude * rand_perlin_2d(shape, (frequency*res[0], frequency*res[1])).unsqueeze(1)
        frequency *= 2
        amplitude *= persistence
    return noise