using Flux

conv_net = Chain(Conv((8,8), 4 => 16, relu, stride=4), 
            Conv((4,4), 16 => 32, relu, stride=2),
            Flux.flatten,
            Dense(2592 => 256, relu), # Assumed that the imput is 84x84x4 as defined in the paper
            Dense(256 => length(𝒜))
            );

            

print(Flux.outputsize(conv_net, (84, 84, 4, 32)))
