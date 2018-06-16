"""
    An example for print model weights
    @author: Liu Weijie
    @date: 2018-04-09
    @reference: <Tensorflow: Deeplearning framework in Action>/Chapter5.4
"""
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# print all variables in checkpoint file
print("=========All Variables==========")
print_tensors_in_checkpoint_file("/home/jagger/workspace/tmp/model.ckpt", tensor_name=None, all_tensors=True, all_tensor_names=True)

# print only tensor v1 in checkpoint file
print("=========V1==========")
print_tensors_in_checkpoint_file("/home/jagger/workspace/tmp/model.ckpt", tensor_name='v1', all_tensors=False, all_tensor_names=False)

# print only tensor v2 in checkpoint file
print("=========V2==========")
print_tensors_in_checkpoint_file("/home/jagger/workspace/tmp/model.ckpt", tensor_name='v2', all_tensors=False, all_tensor_names=False)