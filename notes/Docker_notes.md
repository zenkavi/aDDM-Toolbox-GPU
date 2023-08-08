
To iron out details of Dockerfile

```sh
docker run --rm -it nvidia/cuda:12.2.0-devel-ubuntu22.04 bash
```

To run examples of toolbox

```sh
# Create new file
vim main.cpp 

# Add code from examples

# Compile 
nvcc -o main main.cpp -laddm

# Run
./main
```
