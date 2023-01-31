import torch
print(torch.__version__)

def my_torch_cv(x):
    '''
    Here goes the definition of the CV.

    Inputs:
        x (torch.tensor): input, either scalar or 1-D array
    Return:
        y (torch.tensor): collective variable (scalar)
    '''
    # CV definition
    y = x[:3] - x[3:]

    return torch.cat([torch.sqrt(y.dot(y)).reshape(1), -torch.sqrt(y.dot(y)).reshape(1)])

input_size = 6

# -- DEFINE INPUT -- 
#random 
#x = torch.rand(input_size, dtype=torch.float32, requires_grad=True).unsqueeze(0)
#or by choosing the value(s) of the array
x = torch.tensor([0.]*3 + [1.]*3, dtype=torch.float32, requires_grad=True)

# -- CALCULATE CV -- 
y = my_torch_cv(x).reshape(2)

# -- CALCULATE DERIVATIVES -- 
for yy in y:
    dy = torch.autograd.grad(yy, x, grad_outputs=torch.ones([]), create_graph=True)
    # -- PRINT -- 
    print('CV TEST')
    print('n_input\t: {}'.format(input_size))
    print('x\t: {}'.format(x))
    print('cv\t: {}'.format(yy))
    print('der\t: {}'.format(dy))
    print(dy[0].unsqueeze(1))

# # Compile via tracing
traced_cv   = torch.jit.trace ( my_torch_cv, example_inputs=x )
filename='torch_model.pt'
traced_cv.save(filename)

