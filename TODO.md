[loader](core/loader.py)
1) add parallel loading via multi threading
2) add pre fetching of N+1 batches

[autograd](core/autograd/activations.py)
[autograd](core/autograd/losses.py)
1) Add Cache: instend of recomputing base functions, cache it from forward pass and let backward pass reuse it
