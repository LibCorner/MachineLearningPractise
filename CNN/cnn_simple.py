#conding:utf-8
import numpy as np
'''
卷积
'''
def padding_inputs(inputs,filter_width,filter_height):
    batch,width,height,channel=inputs.shape
    pad_width=(filter_width-1)//2
    pad_height=(filter_height-1)//2
    
    return np.pad(inputs,[(0,0),(pad_width,filter_width-1-pad_width),
                          (pad_height,filter_height-1-pad_height),(0,0)],"constant")

def conv(inputs,filters,padding="SAME",strides=[1,1,1,1]):
    '''
    parameters:
        inputs: 输入,[batch, width,height, channel]
        filters: 卷积核，[input_channel,filter_width,filter_height,output_channel]
        padding: SMAE或VALID
        strides: 移动的步长,[batch,width,height,channel]
    '''
    filter_shape=filters.shape
    assert len(filter_shape)==4
    batch=filter_shape[0]
    filter_width=filter_shape[1]
    filter_height=filter_shape[2]
    output_channel=filter_shape[3]
    if padding=="SAME":
        inputs=padding_inputs(inputs,filter_width,filter_height)
    
    width=inputs.shape[1]
    height=inputs.shape[2]
    outputs=np.zeros([batch,width-filter_width+1,height-filter_height+1,output_channel])
    filters=np.reshape(filters,[np.prod(filters.shape[:3]),filters.shape[3]])
    for i in range(width-filter_width+1):
        for j in range(height-filter_height+1):
            region=inputs[:,i:i+filter_width,j:j+filter_height,:]
            region=np.reshape(region,[batch,np.prod(region.shape[1:])])
            output=np.matmul(region,filters)
            output=np.reshape(output,[batch,1,1,output_channel])
            outputs[:,i,j,:]=output
            
    return outputs
    
    
if __name__=="__main__":
    a=np.array([[1,0,0,0,0,1],
                [0,1,0,0,1,0],
                [0,0,1,1,0,0],
                [1,0,0,0,1,0],
                [0,1,0,0,1,0],
                [0,0,1,0,1,0]])
    a=np.reshape(a,[1,6,6,1])
    filters=np.array([[1,-1,-1],
                      [-1,1,-1],
                      [-1,-1,1]])
    filters=np.reshape(filters,[1,3,3,1])
    outputs=conv(a,filters,padding="SAME")
    outputs=np.reshape(outputs,[6,6])
    print(outputs)
