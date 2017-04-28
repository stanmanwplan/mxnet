# NDArray: NumPy-style Tensor Computations on CPUs and GPUs

`NDArray` is the basic operational unit for matrix and tensor
computations within MXNet. `NDArray` is similar to `numpy.ndarray`, but it supports two very powerful additional
features:

- Multiple devices - Operations can run on different devices, including CPU and GPU cards.
- Automatic parallelization - Operations are automatically executed in parallel when possible.

## Creating and Initializing `NDArray`s

You can create an `NDArray` on either a CPU or a GPU, as follows:

```python
    >>> import mxnet as mx
    >>> a = mx.nd.empty((2, 3)) # Create a 2-by-3 matrix on a CPU
    >>> b = mx.nd.empty((2, 3), mx.gpu()) # Create a 2-by-3 matrix on GPU 0
    >>> c = mx.nd.empty((2, 3), mx.gpu(2)) # Create a 2-by-3 matrix on GPU 2
    >>> c.shape # get shape
    (2L, 3L)
    >>> c.context # get device info
    gpu(2)
```

Each of the `NDArray`s shown in the example above (a, b, and c) have two rows and three columns. The `shape` method returns the length (L) for each dimension. The `context` method returns the operational context for an `NDArray`, such as cpu(0) or gpu(2).  

You can initialize an `NDArray` in multiple ways:

```python
    >>> import mxnet as mx
    >>> a = mx.nd.zeros((2, 3)) # Create a 2-by-3 matrix filled with 0's
    >>> b = mx.nd.ones((2, 3))  # Create a 2-by-3 matrix filled with 1's
    >>> print a.asnumpy()
    [[ 0.  0.  0.]
     [ 0.  0.  0.]]
    >>> print b.asnumpy()
    [[ 1.  1.  1.]
     [ 1.  1.  1.]]
    >>> b[:] = 2 # Set all elements of b to 2.
    >>> print b.asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
```

You can copy the values from one `NDArray` to another, even if the `NDArray`s are located on different devices:

```python
    >>> import mxnet as mx
    >>> a = mx.nd.ones((2, 3))
    >>> print a.asnumpy()
    [[ 1.  1.  1.]
     [ 1.  1.  1.]]
    >>> b = mx.nd.zeros((2, 3), mx.gpu())
    >>> print b.asnumpy()
    [[ 0.  0.  0.]
     [ 0.  0.  0.]]
    >>> a.copyto(b) # Copy the data from a CPU to a GPU
    <NDArray 2x3 @gpu(0)>
    >>> print b.asnumpy()
    [[ 1.  1.  1.]
     [ 1.  1.  1.]]
```

You can convert an `NDArray` to a `numpy.ndarray`:

```python
    >>> import mxnet as mx
    >>> a = mx.nd.ones((2, 3))
    >>> type(a)
    <class 'mxnet.ndarray.NDArray'>
    >>> b = a.asnumpy()
    >>> type(b)
    <type 'numpy.ndarray'>
    >>> print a
    <NDArray 2x3 @cpu(0)>
    >>> print b
    [[ 1.  1.  1.]
    [ 1.  1.  1.]]
```
Note that MXNet `NDArray`s behave differently than `numpy.ndarray`s in some important respects:

- NDArray.T performs real data transpose to return new a copied array, instead of returning a view of the input array.
- NDArray.dot performs a dot operation between the last axis of the first input array and the first axis of the second input array, whereas numpy.dot uses the second last axis of the input array.

You can also convert a `numpy.ndarray` to an `NDArray`:

```python
    >>> import numpy as np
    >>> import mxnet as mx
    >>> a = mx.nd.empty((2, 3))
    >>> print a.asnumpy()
    [[  1.36880317e-06   4.58715052e-41   1.36880317e-06]
     [  4.58715052e-41   1.82058011e-06   4.58715052e-41]]
    >>> a[:] = np.random.uniform(-0.1, 0.1, a.shape)
    >>> print a.asnumpy()
    [[ 0.0371692   0.07489774 -0.02337022]
     [-0.08037076  0.07111147 -0.08474302]]
```

The `empty` method does not fill the `NDArray` with any values. Thus, the `NDArray` will initially contain the pre-existing values at the memory locations where the it was instantiated. 

## Basic Element-wise Operations

By default, `NDArray` methods perform element-wise operations:

```python
    >>> import mxnet as mx
    >>> a = mx.nd.ones((2, 3)) * 2
    >>> print a.asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    >>> b = mx.nd.ones((2, 3)) * 4
    >>> print b.asnumpy()
    [[ 4.  4.  4.]
     [ 4.  4.  4.]]
    >>> c = a + b
    >>> print c.asnumpy()
    [[ 6.  6.  6.]
     [ 6.  6.  6.]]
    >>> d = a * b
    >>> print d.asnumpy()
    [[ 8.  8.  8.]
     [ 8.  8.  8.]]
```

If you wish to perform operations on multiple `NDArray`s which are located on different devices, you must explicitly move the `NDArray`s to the same device. You can do this with the `copy` method. The following example performs computations on GPU 0:

```python
    >>> import mxnet as mx
    >>> a = mx.nd.ones((2, 3)) * 2
    >>> print a.asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    >>> b = mx.nd.ones((2, 3), mx.gpu()) * 3
    >>> print b.asnumpy()
    [[ 3.  3.  3.]
     [ 3.  3.  3.]]
    >>> c = a.copyto(mx.gpu()) * b
    >>> print c.asnumpy()
    [[ 6.  6.  6.]
     [ 6.  6.  6.]]
```

## Loading and Saving `NDArray`s

There are multiple ways that you can save data to files and load data from files. One method is to use
`pickle`.  `NDArray` is compatible with `pickle`, which means that you can simply run `pickle` against the
`NDArray` as you would with `numpy.ndarray`:

 ```python
    >>> import mxnet as mx
    >>> import pickle as pkl
    >>> a = mx.nd.ones((2, 3)) * 2
    >>> print a.asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    >>> data = pkl.dumps(a)
    >>> b = pkl.loads(data)
    >>> print b.asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
 ```
Another method is to directly dump a list of `NDArray`s to a file in binary format:

 ```python
    >>> import mxnet as mx
    >>> a = mx.nd.ones((2,3))*2
    >>> print a.asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    >>> b = mx.nd.ones((2,3))*3
    >>> print b.asnumpy()
    [[ 3.  3.  3.]
     [ 3.  3.  3.]]
    >>> mx.nd.save('mydata.bin', [a, b])
    >>> c = mx.nd.load('mydata.bin')
    >>> print c[0].asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    >>> print c[1].asnumpy()
    [[ 3.  3.  3.]
     [ 3.  3.  3.]]
 ```

It is also possible to dump a dict:

 ```python
    >>> import mxnet as mx
    >>> a = mx.nd.ones((2,3))*2
    >>> print a.asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    >>> b = mx.nd.ones((2,3))*3
    >>> print b.asnumpy()
    [[ 3.  3.  3.]
     [ 3.  3.  3.]]
    >>> mx.nd.save('mydata.bin', {'a':a, 'b':b})
    >>> c = mx.nd.load('mydata.bin')
    >>> print c['a'].asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    >>> print c['b'].asnumpy()
    [[ 3.  3.  3.]
     [ 3.  3.  3.]]
 ```

In addition, if you have set up distributed file systems, such as Amazon S3 and HDFS, you
can directly save to and load data from those file systems:

 ```python
    >>> import mxnet as mx
    >>> a = mx.nd.ones((2,3))*2
    >>> b = mx.nd.ones((2,3))*3
    >>> mx.nd.save('s3://mybucket/mydata.bin', [a,b])
    >>> mx.nd.save('hdfs://users/myname/mydata.bin', [a,b])
 ```

## Automatic Parallelization
An `NDArray` can automatically execute operations in parallel. This is very helpful when you
use multiple compute resources, such as CPU and GPU cards, along with CPU-to-GPU memory bandwidth.

For example, if your code includes the statement `a += 1` followed by `b += 1`, and `a` is on a CPU card while
`b` is on a GPU card, then you will want to execute those operations in parallel in order to maximize performance. In addition, data copies between CPU and GPU are expensive, so you will
want to run them in parallel with other computations.

However, it is difficult to manually determine which statements can be executed in parallel. In the
following example, `a+=1` and `c*=3` can be executed in parallel, but `a+=1` and
`b*=3` must be executed sequentially.

 ```python
    >>> import mxnet as mx
    >>> a = mx.nd.ones((2,3))
    >>> b = a
    >>> c = a.copyto(mx.cpu())
    >>> a += 1
    >>> b *= 3
    >>> c *= 3
 ```

Fortunately, MXNet can automatically resolve the dependencies and
execute operations in parallel while guaranteeing correct execution. Because of this, you
can write a program as if it is using only a single thread, and MXNet will
automatically dispatch it to multiple devices when possible, including multiple GPU cards and multiple
computers.

MXNet achieves this by using lazy evaluation. Any operation in the code is issued to an
internal engine, and then returned. For example, if you run `a += 1`, it
returns immediately after pushing the plus operation to the engine. This
asynchronism allows MXNet to push more operations to the engine, so it can determine
the read and write dependencies and determine the best way to execute operations in
parallel.

The actual computations are completed when the code copies the results to another location, for example when you call `print a.asnumpy()` or `mx.nd.save([a])`. Thus, to achieve high levels of parallelization, your code should postpone retrieving
the results until after all computations are completed.

##  Next Steps
* [Symbol](symbol.md)
* [KVStore](kvstore.md)
