ЦВ

5л4
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetype
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Р
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

П
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
ы
FusedBatchNormGradV3

y_backprop"T
x"T	
scale
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U

x_backprop"T
scale_backprop"U
offset_backprop"U
reserve_space_4"U
reserve_space_5"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Зб8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
Ф
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Зб8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
д
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ю
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
р
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.15.02v1.15.0-rc3-22-g590d6eef7eЦЅ	

conv2d_1_inputPlaceholder*
dtype0*1
_output_shapes
:џџџџџџџџџрр*&
shape:џџџџџџџџџрр
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"         `   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:

.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *еVЗМ*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *еVЗ<*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
і
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:`*

seed *
T0*"
_class
loc:@conv2d_1/kernel*
seed2 
к
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
є
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*&
_output_shapes
:`*
T0*"
_class
loc:@conv2d_1/kernel
ц
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*&
_output_shapes
:`*
T0*"
_class
loc:@conv2d_1/kernel
З
conv2d_1/kernelVarHandleOp*
shape:`*
dtype0*
_output_shapes
: * 
shared_nameconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
	container 
o
0conv2d_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
t
conv2d_1/kernel/AssignAssignVariableOpconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:`

conv2d_1/bias/Initializer/zerosConst*
valueB`*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:`
Ѕ
conv2d_1/biasVarHandleOp*
	container *
shape:`*
dtype0*
_output_shapes
: *
shared_nameconv2d_1/bias* 
_class
loc:@conv2d_1/bias
k
.conv2d_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
e
conv2d_1/bias/AssignAssignVariableOpconv2d_1/biasconv2d_1/bias/Initializer/zeros*
dtype0
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:`
g
conv2d_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:`

conv2d_1/Conv2DConv2Dconv2d_1_inputconv2d_1/Conv2D/ReadVariableOp*/
_output_shapes
:џџџџџџџџџ66`*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
i
conv2d_1/BiasAdd/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:`

conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ66`
a
conv2d_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ66`
О
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџ`
Љ
,batch_normalization_1/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:`*
valueB`*  ?*.
_class$
" loc:@batch_normalization_1/gamma
Я
batch_normalization_1/gammaVarHandleOp*.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:`*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_1/gamma

<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 

"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*
dtype0

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes
:`
Ј
,batch_normalization_1/beta/Initializer/zerosConst*
valueB`*    *-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
:`
Ь
batch_normalization_1/betaVarHandleOp*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:`

;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 

!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*
dtype0

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes
:`
Ж
3batch_normalization_1/moving_mean/Initializer/zerosConst*
valueB`*    *4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:`
с
!batch_normalization_1/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container *
shape:`

Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
Ё
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*
dtype0

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:`
Н
6batch_normalization_1/moving_variance/Initializer/onesConst*
valueB`*  ?*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:`
э
%batch_normalization_1/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_1/moving_variance*8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape:`

Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
Ќ
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*
dtype0

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:`
|
$batch_normalization_1/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes
:`
}
&batch_normalization_1/ReadVariableOp_1ReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes
:`

5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:`

7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:`
Ќ
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3max_pooling2d_1/MaxPool$batch_normalization_1/ReadVariableOp&batch_normalization_1/ReadVariableOp_15batch_normalization_1/FusedBatchNormV3/ReadVariableOp7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
T0*
U0*
data_formatNHWC*
is_training( *
epsilon%o:*K
_output_shapes9
7:џџџџџџџџџ`:`:`:`:`:
`
batch_normalization_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?
­
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*%
valueB"      `      *"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
:

.conv2d_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *шеМ*"
_class
loc:@conv2d_2/kernel

.conv2d_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ше<*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 
ї
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*

seed *
T0*"
_class
loc:@conv2d_2/kernel*
seed2 *
dtype0*'
_output_shapes
:`
к
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
ѕ
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:`
ч
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*'
_output_shapes
:`*
T0*"
_class
loc:@conv2d_2/kernel
И
conv2d_2/kernelVarHandleOp* 
shared_nameconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
	container *
shape:`*
dtype0*
_output_shapes
: 
o
0conv2d_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
t
conv2d_2/kernel/AssignAssignVariableOpconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
dtype0
|
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*'
_output_shapes
:`

conv2d_2/bias/Initializer/zerosConst*
valueB*    * 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes	
:
І
conv2d_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
	container *
shape:
k
.conv2d_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
e
conv2d_2/bias/AssignAssignVariableOpconv2d_2/biasconv2d_2/bias/Initializer/zeros*
dtype0
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes	
:
g
conv2d_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
w
conv2d_2/Conv2D/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*'
_output_shapes
:`
Є
conv2d_2/Conv2DConv2D&batch_normalization_1/FusedBatchNormV3conv2d_2/Conv2D/ReadVariableOp*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0
j
conv2d_2/BiasAdd/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes	
:

conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџ
b
conv2d_2/ReluReluconv2d_2/BiasAdd*
T0*0
_output_shapes
:џџџџџџџџџ
П
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*0
_output_shapes
:џџџџџџџџџ

*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
Ћ
,batch_normalization_2/gamma/Initializer/onesConst*
valueB*  ?*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes	
:
а
batch_normalization_2/gammaVarHandleOp*,
shared_namebatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:*
dtype0*
_output_shapes
: 

<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 

"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
dtype0

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes	
:
Њ
,batch_normalization_2/beta/Initializer/zerosConst*
valueB*    *-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes	
:
Э
batch_normalization_2/betaVarHandleOp*+
shared_namebatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:*
dtype0*
_output_shapes
: 

;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 

!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*
dtype0

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes	
:
И
3batch_normalization_2/moving_mean/Initializer/zerosConst*
valueB*    *4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:
т
!batch_normalization_2/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:

Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
Ё
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*
dtype0

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:
П
6batch_normalization_2/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*8
_class.
,*loc:@batch_normalization_2/moving_variance
ю
%batch_normalization_2/moving_varianceVarHandleOp*8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_2/moving_variance

Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
Ќ
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*
dtype0

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:
}
$batch_normalization_2/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes	
:
~
&batch_normalization_2/ReadVariableOp_1ReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes	
:

5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:

7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:
Б
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3max_pooling2d_2/MaxPool$batch_normalization_2/ReadVariableOp&batch_normalization_2/ReadVariableOp_15batch_normalization_2/FusedBatchNormV3/ReadVariableOp7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
T0*
U0*
data_formatNHWC*
is_training( *
epsilon%o:*P
_output_shapes>
<:џџџџџџџџџ

:::::
`
batch_normalization_2/ConstConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
u
flatten_1/ShapeShape&batch_normalization_2/FusedBatchNormV3*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
i
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ћ
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
flatten_1/Reshape/shape/1Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

flatten_1/Reshape/shapePackflatten_1/strided_sliceflatten_1/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 

flatten_1/ReshapeReshape&batch_normalization_2/FusedBatchNormV3flatten_1/Reshape/shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџШ
Ѓ
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB" d     *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *BуhМ*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *Bуh<*!
_class
loc:@dense_1/kernel
ю
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*!
_output_shapes
:Ш *

seed *
T0*!
_class
loc:@dense_1/kernel*
seed2 
ж
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
ы
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:Ш 
н
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:Ш 
Џ
dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
	container *
shape:Ш 
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
t
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*!
_output_shapes
:Ш 

.dense_1/bias/Initializer/zeros/shape_as_tensorConst*
valueB: *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:

$dense_1/bias/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
е
dense_1/bias/Initializer/zerosFill.dense_1/bias/Initializer/zeros/shape_as_tensor$dense_1/bias/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense_1/bias*
_output_shapes	
: 
Ѓ
dense_1/biasVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
	container 
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
: 
o
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*!
_output_shapes
:Ш 
Ѓ
dense_1/MatMulMatMulflatten_1/Reshapedense_1/MatMul/ReadVariableOp*(
_output_shapes
:џџџџџџџџџ *
transpose_a( *
transpose_b( *
T0
h
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
: 

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ *
T0
X
dense_1/TanhTanhdense_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ 
_
dropout_1/IdentityIdentitydense_1/Tanh*
T0*(
_output_shapes
:џџџџџџџџџ 
Ѓ
/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
:

-dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *зГнМ*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 

-dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *зГн<*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 
э
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
  *

seed *
T0*!
_class
loc:@dense_2/kernel
ж
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
ъ
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
  
м
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
  
Ў
dense_2/kernelVarHandleOp*
	container *
shape:
  *
dtype0*
_output_shapes
: *
shared_namedense_2/kernel*!
_class
loc:@dense_2/kernel
m
/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/kernel*
_output_shapes
: 
q
dense_2/kernel/AssignAssignVariableOpdense_2/kernel)dense_2/kernel/Initializer/random_uniform*
dtype0
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
  

.dense_2/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB: *
_class
loc:@dense_2/bias

$dense_2/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@dense_2/bias
е
dense_2/bias/Initializer/zerosFill.dense_2/bias/Initializer/zeros/shape_as_tensor$dense_2/bias/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense_2/bias*
_output_shapes	
: 
Ѓ
dense_2/biasVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_namedense_2/bias*
_class
loc:@dense_2/bias*
	container 
i
-dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/bias*
_output_shapes
: 
b
dense_2/bias/AssignAssignVariableOpdense_2/biasdense_2/bias/Initializer/zeros*
dtype0
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
: 
n
dense_2/MatMul/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
  
Є
dense_2/MatMulMatMuldropout_1/Identitydense_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ *
transpose_a( *
transpose_b( 
h
dense_2/BiasAdd/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
: 

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ 
X
dense_2/TanhTanhdense_2/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ 
_
dropout_2/IdentityIdentitydense_2/Tanh*
T0*(
_output_shapes
:џџџџџџџџџ 
Ѓ
/dense_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *!
_class
loc:@dense_3/kernel

-dense_3/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *IvН*!
_class
loc:@dense_3/kernel

-dense_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *Iv=*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
ь
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	 *

seed *
T0*!
_class
loc:@dense_3/kernel*
seed2 
ж
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_3/kernel
щ
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
:	 
л
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
:	 
­
dense_3/kernelVarHandleOp*!
_class
loc:@dense_3/kernel*
	container *
shape:	 *
dtype0*
_output_shapes
: *
shared_namedense_3/kernel
m
/dense_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/kernel*
_output_shapes
: 
q
dense_3/kernel/AssignAssignVariableOpdense_3/kernel)dense_3/kernel/Initializer/random_uniform*
dtype0
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0*
_output_shapes
:	 

dense_3/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
:
Ђ
dense_3/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_3/bias*
_class
loc:@dense_3/bias*
	container *
shape:
i
-dense_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/bias*
_output_shapes
: 
b
dense_3/bias/AssignAssignVariableOpdense_3/biasdense_3/bias/Initializer/zeros*
dtype0
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:
m
dense_3/MatMul/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0*
_output_shapes
:	 
Ѓ
dense_3/MatMulMatMuldropout_2/Identitydense_3/MatMul/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
g
dense_3/BiasAdd/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
]
dense_3/SoftmaxSoftmaxdense_3/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
Д
PlaceholderPlaceholder*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
O
AssignVariableOpAssignVariableOpconv2d_1/kernelPlaceholder*
dtype0
y
ReadVariableOpReadVariableOpconv2d_1/kernel^AssignVariableOp*
dtype0*&
_output_shapes
:`
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
Q
AssignVariableOp_1AssignVariableOpconv2d_1/biasPlaceholder_1*
dtype0
o
ReadVariableOp_1ReadVariableOpconv2d_1/bias^AssignVariableOp_1*
dtype0*
_output_shapes
:`
h
Placeholder_2Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
_
AssignVariableOp_2AssignVariableOpbatch_normalization_1/gammaPlaceholder_2*
dtype0
}
ReadVariableOp_2ReadVariableOpbatch_normalization_1/gamma^AssignVariableOp_2*
dtype0*
_output_shapes
:`
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
^
AssignVariableOp_3AssignVariableOpbatch_normalization_1/betaPlaceholder_3*
dtype0
|
ReadVariableOp_3ReadVariableOpbatch_normalization_1/beta^AssignVariableOp_3*
dtype0*
_output_shapes
:`
h
Placeholder_4Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
e
AssignVariableOp_4AssignVariableOp!batch_normalization_1/moving_meanPlaceholder_4*
dtype0

ReadVariableOp_4ReadVariableOp!batch_normalization_1/moving_mean^AssignVariableOp_4*
dtype0*
_output_shapes
:`
h
Placeholder_5Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
i
AssignVariableOp_5AssignVariableOp%batch_normalization_1/moving_variancePlaceholder_5*
dtype0

ReadVariableOp_5ReadVariableOp%batch_normalization_1/moving_variance^AssignVariableOp_5*
dtype0*
_output_shapes
:`
Ж
Placeholder_6Placeholder*
dtype0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
S
AssignVariableOp_6AssignVariableOpconv2d_2/kernelPlaceholder_6*
dtype0
~
ReadVariableOp_6ReadVariableOpconv2d_2/kernel^AssignVariableOp_6*
dtype0*'
_output_shapes
:`
h
Placeholder_7Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
Q
AssignVariableOp_7AssignVariableOpconv2d_2/biasPlaceholder_7*
dtype0
p
ReadVariableOp_7ReadVariableOpconv2d_2/bias^AssignVariableOp_7*
dtype0*
_output_shapes	
:
h
Placeholder_8Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
_
AssignVariableOp_8AssignVariableOpbatch_normalization_2/gammaPlaceholder_8*
dtype0
~
ReadVariableOp_8ReadVariableOpbatch_normalization_2/gamma^AssignVariableOp_8*
dtype0*
_output_shapes	
:
h
Placeholder_9Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
^
AssignVariableOp_9AssignVariableOpbatch_normalization_2/betaPlaceholder_9*
dtype0
}
ReadVariableOp_9ReadVariableOpbatch_normalization_2/beta^AssignVariableOp_9*
dtype0*
_output_shapes	
:
i
Placeholder_10Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
g
AssignVariableOp_10AssignVariableOp!batch_normalization_2/moving_meanPlaceholder_10*
dtype0

ReadVariableOp_10ReadVariableOp!batch_normalization_2/moving_mean^AssignVariableOp_10*
dtype0*
_output_shapes	
:
i
Placeholder_11Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
k
AssignVariableOp_11AssignVariableOp%batch_normalization_2/moving_variancePlaceholder_11*
dtype0

ReadVariableOp_11ReadVariableOp%batch_normalization_2/moving_variance^AssignVariableOp_11*
dtype0*
_output_shapes	
:

Placeholder_12Placeholder*%
shape:џџџџџџџџџџџџџџџџџџ*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
T
AssignVariableOp_12AssignVariableOpdense_1/kernelPlaceholder_12*
dtype0
y
ReadVariableOp_12ReadVariableOpdense_1/kernel^AssignVariableOp_12*
dtype0*!
_output_shapes
:Ш 
i
Placeholder_13Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
R
AssignVariableOp_13AssignVariableOpdense_1/biasPlaceholder_13*
dtype0
q
ReadVariableOp_13ReadVariableOpdense_1/bias^AssignVariableOp_13*
dtype0*
_output_shapes	
: 

Placeholder_14Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
T
AssignVariableOp_14AssignVariableOpdense_2/kernelPlaceholder_14*
dtype0
x
ReadVariableOp_14ReadVariableOpdense_2/kernel^AssignVariableOp_14*
dtype0* 
_output_shapes
:
  
i
Placeholder_15Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
R
AssignVariableOp_15AssignVariableOpdense_2/biasPlaceholder_15*
dtype0
q
ReadVariableOp_15ReadVariableOpdense_2/bias^AssignVariableOp_15*
dtype0*
_output_shapes	
: 

Placeholder_16Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
T
AssignVariableOp_16AssignVariableOpdense_3/kernelPlaceholder_16*
dtype0
w
ReadVariableOp_16ReadVariableOpdense_3/kernel^AssignVariableOp_16*
dtype0*
_output_shapes
:	 
i
Placeholder_17Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
R
AssignVariableOp_17AssignVariableOpdense_3/biasPlaceholder_17*
dtype0
p
ReadVariableOp_17ReadVariableOpdense_3/bias^AssignVariableOp_17*
dtype0*
_output_shapes
:
g
VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
P
VarIsInitializedOp_1VarIsInitializedOpdense_1/bias*
_output_shapes
: 
R
VarIsInitializedOp_2VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
e
VarIsInitializedOp_3VarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
S
VarIsInitializedOp_4VarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
e
VarIsInitializedOp_5VarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
Q
VarIsInitializedOp_6VarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
^
VarIsInitializedOp_7VarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
_
VarIsInitializedOp_8VarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
i
VarIsInitializedOp_9VarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
T
VarIsInitializedOp_10VarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
R
VarIsInitializedOp_11VarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
`
VarIsInitializedOp_12VarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
_
VarIsInitializedOp_13VarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
S
VarIsInitializedOp_14VarIsInitializedOpdense_2/kernel*
_output_shapes
: 
S
VarIsInitializedOp_15VarIsInitializedOpdense_3/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_16VarIsInitializedOpdense_3/bias*
_output_shapes
: 
Q
VarIsInitializedOp_17VarIsInitializedOpdense_2/bias*
_output_shapes
: 
М
initNoOp"^batch_normalization_1/beta/Assign#^batch_normalization_1/gamma/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign"^batch_normalization_2/beta/Assign#^batch_normalization_2/gamma/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign

dense_3_targetPlaceholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
v
total/Initializer/zerosConst*
valueB
 *    *
_class

loc:@total*
dtype0*
_output_shapes
: 

totalVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_nametotal*
_class

loc:@total*
	container 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
M
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
valueB
 *    *
_class

loc:@count*
dtype0*
_output_shapes
: 

countVarHandleOp*
shared_namecount*
_class

loc:@count*
	container *
shape: *
dtype0*
_output_shapes
: 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
M
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

metrics/acc/ArgMaxArgMaxdense_3_targetmetrics/acc/ArgMax/dimension*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

metrics/acc/ArgMax_1ArgMaxdense_3/Softmaxmetrics/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0

metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
incompatible_shape_error(*
T0	*#
_output_shapes
:џџџџџџџџџ
x
metrics/acc/CastCastmetrics/acc/Equal*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0

[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
y
metrics/acc/SumSummetrics/acc/Castmetrics/acc/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0

metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
dtype0*
_output_shapes
: 
[
metrics/acc/SizeSizemetrics/acc/Cast*
T0*
out_type0*
_output_shapes
: 
l
metrics/acc/Cast_1Castmetrics/acc/Size*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_1 ^metrics/acc/AssignAddVariableOp*
dtype0
 
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
T0*
_output_shapes
: 
\
loss/dense_3_loss/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
8loss/dense_3_loss/softmax_cross_entropy_with_logits/RankConst*
dtype0*
_output_shapes
: *
value	B :

9loss/dense_3_loss/softmax_cross_entropy_with_logits/ShapeShapedense_3/BiasAdd*
T0*
out_type0*
_output_shapes
:
|
:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :

;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_1Shapedense_3/BiasAdd*
_output_shapes
:*
T0*
out_type0
{
9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
ж
7loss/dense_3_loss/softmax_cross_entropy_with_logits/SubSub:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_19loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
К
?loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/beginPack7loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N*
_output_shapes
:

>loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
В
9loss/dense_3_loss/softmax_cross_entropy_with_logits/SliceSlice;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_1?loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/begin>loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:

Closs/dense_3_loss/softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:

?loss/dense_3_loss/softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
С
:loss/dense_3_loss/softmax_cross_entropy_with_logits/concatConcatV2Closs/dense_3_loss/softmax_cross_entropy_with_logits/concat/values_09loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice?loss/dense_3_loss/softmax_cross_entropy_with_logits/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
м
;loss/dense_3_loss/softmax_cross_entropy_with_logits/ReshapeReshapedense_3/BiasAdd:loss/dense_3_loss/softmax_cross_entropy_with_logits/concat*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0
|
:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :

;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_2Shapedense_3_target*
T0*
out_type0*
_output_shapes
:
}
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
к
9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1Sub:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_2;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
О
Aloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/beginPack9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1*
T0*

axis *
N*
_output_shapes
:

@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
И
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1Slice;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_2Aloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/begin@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:

Eloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:

Aloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Щ
<loss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1ConcatV2Eloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/values_0;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1Aloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
п
=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_1Reshapedense_3_target<loss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

3loss/dense_3_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits;loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_1*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
}
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
и
9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2Sub8loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 

Aloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Н
@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/sizePack9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N*
_output_shapes
:
Ж
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2Slice9loss/dense_3_loss/softmax_cross_entropy_with_logits/ShapeAloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/begin@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/size*
Index0*
T0*
_output_shapes
:
і
=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2Reshape3loss/dense_3_loss/softmax_cross_entropy_with_logits;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
k
&loss/dense_3_loss/weighted_loss/Cast/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Tloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 

Sloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
а
Sloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2*
_output_shapes
:*
T0*
out_type0

Rloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
j
bloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
Ѓ
Aloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2c^loss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
ы
Aloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/ConstConstc^loss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  ?

;loss/dense_3_loss/weighted_loss/broadcast_weights/ones_likeFillAloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeAloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:џџџџџџџџџ
Ы
1loss/dense_3_loss/weighted_loss/broadcast_weightsMul&loss/dense_3_loss/weighted_loss/Cast/x;loss/dense_3_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:џџџџџџџџџ
Ъ
#loss/dense_3_loss/weighted_loss/MulMul=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_21loss/dense_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:џџџџџџџџџ
c
loss/dense_3_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_3_loss/SumSum#loss/dense_3_loss/weighted_loss/Mulloss/dense_3_loss/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
|
loss/dense_3_loss/num_elementsSize#loss/dense_3_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
: 

#loss/dense_3_loss/num_elements/CastCastloss/dense_3_loss/num_elements*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
\
loss/dense_3_loss/Const_2Const*
valueB *
dtype0*
_output_shapes
: 

loss/dense_3_loss/Sum_1Sumloss/dense_3_loss/Sumloss/dense_3_loss/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

loss/dense_3_loss/valueDivNoNanloss/dense_3_loss/Sum_1#loss/dense_3_loss/num_elements/Cast*
_output_shapes
: *
T0
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/dense_3_loss/value*
_output_shapes
: *
T0
j
'training/Adam/gradients/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
p
+training/Adam/gradients/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
З
&training/Adam/gradients/gradients/FillFill'training/Adam/gradients/gradients/Shape+training/Adam/gradients/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 

3training/Adam/gradients/gradients/loss/mul_grad/MulMul&training/Adam/gradients/gradients/Fillloss/dense_3_loss/value*
T0*
_output_shapes
: 

5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Mul&training/Adam/gradients/gradients/Fill
loss/mul/x*
T0*
_output_shapes
: 

Dtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

Ftraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
И
Ttraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsDtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/ShapeFtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
в
Itraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nanDivNoNan5training/Adam/gradients/gradients/loss/mul_grad/Mul_1#loss/dense_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
Ј
Btraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/SumSumItraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nanTtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/BroadcastGradientArgs*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

Ftraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/ReshapeReshapeBtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/SumDtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Btraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/NegNegloss/dense_3_loss/Sum_1*
T0*
_output_shapes
: 
с
Ktraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nan_1DivNoNanBtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Neg#loss/dense_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
ъ
Ktraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nan_2DivNoNanKtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nan_1#loss/dense_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
ю
Btraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/mulMul5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Ktraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
Ѕ
Dtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Sum_1SumBtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/mulVtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

Htraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Reshape_1ReshapeDtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Sum_1Ftraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

Ltraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

Ftraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/ReshapeReshapeFtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/ReshapeLtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 

Dtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 

Ctraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/TileTileFtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/ReshapeDtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/Const*
T0*
_output_shapes
: *

Tmultiples0

Jtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

Dtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/ReshapeReshapeCtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/TileJtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
Ѕ
Btraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/ShapeShape#loss/dense_3_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
:

Atraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/TileTileDtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/ReshapeBtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/Shape*
T0*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0
Э
Ptraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/ShapeShape=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:
У
Rtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Shape_1Shape1loss/dense_3_loss/weighted_loss/broadcast_weights*
T0*
out_type0*
_output_shapes
:
м
`training/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/ShapeRtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
љ
Ntraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/MulMulAtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/Tile1loss/dense_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:џџџџџџџџџ
Ч
Ntraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/SumSumNtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Mul`training/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
Rtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/ReshapeReshapeNtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/SumPtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

Ptraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Mul_1Mul=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2Atraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/Tile*
T0*#
_output_shapes
:џџџџџџџџџ
Э
Ptraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Sum_1SumPtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Mul_1btraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
Ttraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Reshape_1ReshapePtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Sum_1Rtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
н
jtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape3loss/dense_3_loss/softmax_cross_entropy_with_logits*
T0*
out_type0*
_output_shapes
:
ѓ
ltraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapeRtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Reshapejtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
Ћ
,training/Adam/gradients/gradients/zeros_like	ZerosLike5loss/dense_3_loss/softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Д
itraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

etraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsltraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapeitraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
О
^training/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/mulMuletraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims5loss/dense_3_loss/softmax_cross_entropy_with_logits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
ы
etraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax;loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

^training/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/NegNegetraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ж
ktraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

gtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsltraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapektraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*
T0*'
_output_shapes
:џџџџџџџџџ*

Tdim0
ы
`training/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/mul_1Mulgtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1^training/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/Neg*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
З
htraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapedense_3/BiasAdd*
_output_shapes
:*
T0*
out_type0
џ
jtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshape^training/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/mulhtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
љ
Btraining/Adam/gradients/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGradjtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
data_formatNHWC*
_output_shapes
:*
T0
Њ
<training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMulMatMuljtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshapedense_3/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ *
transpose_a( *
transpose_b(

>training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMul_1MatMuldropout_2/Identityjtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*
_output_shapes
:	 *
transpose_a(*
transpose_b( 
Ч
<training/Adam/gradients/gradients/dense_2/Tanh_grad/TanhGradTanhGraddense_2/Tanh<training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMul*
T0*(
_output_shapes
:џџџџџџџџџ 
Ь
Btraining/Adam/gradients/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/gradients/dense_2/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
: *
T0
ќ
<training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMulMatMul<training/Adam/gradients/gradients/dense_2/Tanh_grad/TanhGraddense_2/MatMul/ReadVariableOp*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ *
transpose_a( 
ы
>training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMul_1MatMuldropout_1/Identity<training/Adam/gradients/gradients/dense_2/Tanh_grad/TanhGrad*
T0* 
_output_shapes
:
  *
transpose_a(*
transpose_b( 
Ч
<training/Adam/gradients/gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanh<training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMul*
T0*(
_output_shapes
:џџџџџџџџџ 
Ь
Btraining/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/gradients/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
: 
§
<training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMulMatMul<training/Adam/gradients/gradients/dense_1/Tanh_grad/TanhGraddense_1/MatMul/ReadVariableOp*
T0*)
_output_shapes
:џџџџџџџџџШ*
transpose_a( *
transpose_b(
ы
>training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1MatMulflatten_1/Reshape<training/Adam/gradients/gradients/dense_1/Tanh_grad/TanhGrad*
T0*!
_output_shapes
:Ш *
transpose_a(*
transpose_b( 
Є
>training/Adam/gradients/gradients/flatten_1/Reshape_grad/ShapeShape&batch_normalization_2/FusedBatchNormV3*
T0*
out_type0*
_output_shapes
:

@training/Adam/gradients/gradients/flatten_1/Reshape_grad/ReshapeReshape<training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul>training/Adam/gradients/gradients/flatten_1/Reshape_grad/Shape*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџ



.training/Adam/gradients/gradients/zeros_like_1	ZerosLike(batch_normalization_2/FusedBatchNormV3:1*
_output_shapes	
:*
T0

.training/Adam/gradients/gradients/zeros_like_2	ZerosLike(batch_normalization_2/FusedBatchNormV3:2*
T0*
_output_shapes	
:

.training/Adam/gradients/gradients/zeros_like_3	ZerosLike(batch_normalization_2/FusedBatchNormV3:3*
T0*
_output_shapes	
:

.training/Adam/gradients/gradients/zeros_like_4	ZerosLike(batch_normalization_2/FusedBatchNormV3:4*
_output_shapes	
:*
T0

.training/Adam/gradients/gradients/zeros_like_5	ZerosLike(batch_normalization_2/FusedBatchNormV3:5*
T0*
_output_shapes
:
Б
btraining/Adam/gradients/gradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3FusedBatchNormGradV3@training/Adam/gradients/gradients/flatten_1/Reshape_grad/Reshapemax_pooling2d_2/MaxPool$batch_normalization_2/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1(batch_normalization_2/FusedBatchNormV3:5*
T0*
U0*
data_formatNHWC*
is_training( *
epsilon%o:*L
_output_shapes:
8:џџџџџџџџџ

::::
ѓ
Jtraining/Adam/gradients/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_2/Relumax_pooling2d_2/MaxPoolbtraining/Adam/gradients/gradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3*0
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
п
=training/Adam/gradients/gradients/conv2d_2/Relu_grad/ReluGradReluGradJtraining/Adam/gradients/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_2/Relu*0
_output_shapes
:џџџџџџџџџ*
T0
Ю
Ctraining/Adam/gradients/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad=training/Adam/gradients/gradients/conv2d_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
г
=training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeNShapeN&batch_normalization_1/FusedBatchNormV3conv2d_2/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
С
Jtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_2/Conv2D/ReadVariableOp=training/Adam/gradients/gradients/conv2d_2/Relu_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ`*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
Х
Ktraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter&batch_normalization_1/FusedBatchNormV3?training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeN:1=training/Adam/gradients/gradients/conv2d_2/Relu_grad/ReluGrad*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*'
_output_shapes
:`

.training/Adam/gradients/gradients/zeros_like_6	ZerosLike(batch_normalization_1/FusedBatchNormV3:1*
T0*
_output_shapes
:`

.training/Adam/gradients/gradients/zeros_like_7	ZerosLike(batch_normalization_1/FusedBatchNormV3:2*
T0*
_output_shapes
:`

.training/Adam/gradients/gradients/zeros_like_8	ZerosLike(batch_normalization_1/FusedBatchNormV3:3*
T0*
_output_shapes
:`

.training/Adam/gradients/gradients/zeros_like_9	ZerosLike(batch_normalization_1/FusedBatchNormV3:4*
T0*
_output_shapes
:`

/training/Adam/gradients/gradients/zeros_like_10	ZerosLike(batch_normalization_1/FusedBatchNormV3:5*
T0*
_output_shapes
:
Ж
btraining/Adam/gradients/gradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3FusedBatchNormGradV3Jtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputmax_pooling2d_1/MaxPool$batch_normalization_1/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1(batch_normalization_1/FusedBatchNormV3:5*G
_output_shapes5
3:џџџџџџџџџ`:`:`:`:`*
T0*
U0*
data_formatNHWC*
is_training( *
epsilon%o:
ђ
Jtraining/Adam/gradients/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_1/Relumax_pooling2d_1/MaxPoolbtraining/Adam/gradients/gradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3*/
_output_shapes
:џџџџџџџџџ66`*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
о
=training/Adam/gradients/gradients/conv2d_1/Relu_grad/ReluGradReluGradJtraining/Adam/gradients/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradconv2d_1/Relu*/
_output_shapes
:џџџџџџџџџ66`*
T0
Э
Ctraining/Adam/gradients/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad=training/Adam/gradients/gradients/conv2d_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:`
Л
=training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeNShapeNconv2d_1_inputconv2d_1/Conv2D/ReadVariableOp*
N* 
_output_shapes
::*
T0*
out_type0
У
Jtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeNconv2d_1/Conv2D/ReadVariableOp=training/Adam/gradients/gradients/conv2d_1/Relu_grad/ReluGrad*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*1
_output_shapes
:џџџџџџџџџрр
Ќ
Ktraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_1_input?training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeN:1=training/Adam/gradients/gradients/conv2d_1/Relu_grad/ReluGrad*&
_output_shapes
:`*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID

$training/Adam/iter/Initializer/zerosConst*
value	B	 R *%
_class
loc:@training/Adam/iter*
dtype0	*
_output_shapes
: 
А
training/Adam/iterVarHandleOp*#
shared_nametraining/Adam/iter*%
_class
loc:@training/Adam/iter*
	container *
shape: *
dtype0	*
_output_shapes
: 
u
3training/Adam/iter/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/iter*
_output_shapes
: 
t
training/Adam/iter/AssignAssignVariableOptraining/Adam/iter$training/Adam/iter/Initializer/zeros*
dtype0	
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 

.training/Adam/beta_1/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*'
_class
loc:@training/Adam/beta_1
Ж
training/Adam/beta_1VarHandleOp*
shape: *
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/beta_1*'
_class
loc:@training/Adam/beta_1*
	container 
y
5training/Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 

training/Adam/beta_1/AssignAssignVariableOptraining/Adam/beta_1.training/Adam/beta_1/Initializer/initial_value*
dtype0
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 

.training/Adam/beta_2/Initializer/initial_valueConst*
valueB
 *wО?*'
_class
loc:@training/Adam/beta_2*
dtype0*
_output_shapes
: 
Ж
training/Adam/beta_2VarHandleOp*
shape: *
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/beta_2*'
_class
loc:@training/Adam/beta_2*
	container 
y
5training/Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 

training/Adam/beta_2/AssignAssignVariableOptraining/Adam/beta_2.training/Adam/beta_2/Initializer/initial_value*
dtype0
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 

-training/Adam/decay/Initializer/initial_valueConst*
valueB
 *    *&
_class
loc:@training/Adam/decay*
dtype0*
_output_shapes
: 
Г
training/Adam/decayVarHandleOp*
shape: *
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/decay*&
_class
loc:@training/Adam/decay*
	container 
w
4training/Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/decay*
_output_shapes
: 

training/Adam/decay/AssignAssignVariableOptraining/Adam/decay-training/Adam/decay/Initializer/initial_value*
dtype0
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
dtype0*
_output_shapes
: 
Њ
5training/Adam/learning_rate/Initializer/initial_valueConst*
valueB
 *o:*.
_class$
" loc:@training/Adam/learning_rate*
dtype0*
_output_shapes
: 
Ы
training/Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *,
shared_nametraining/Adam/learning_rate*.
_class$
" loc:@training/Adam/learning_rate*
	container *
shape: 

<training/Adam/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 

"training/Adam/learning_rate/AssignAssignVariableOptraining/Adam/learning_rate5training/Adam/learning_rate/Initializer/initial_value*
dtype0

/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
О
Atraining/Adam/conv2d_1/kernel/m/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_1/kernel*%
valueB"         `   *
dtype0*
_output_shapes
:
 
7training/Adam/conv2d_1/kernel/m/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/conv2d_1/kernel/m/Initializer/zerosFillAtraining/Adam/conv2d_1/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_1/kernel/m/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_1/kernel*

index_type0*&
_output_shapes
:`
з
training/Adam/conv2d_1/kernel/mVarHandleOp*0
shared_name!training/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
	container *
shape:`*
dtype0*
_output_shapes
: 
Г
@training/Adam/conv2d_1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 

&training/Adam/conv2d_1/kernel/m/AssignAssignVariableOptraining/Adam/conv2d_1/kernel/m1training/Adam/conv2d_1/kernel/m/Initializer/zeros*
dtype0
П
3training/Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:`

/training/Adam/conv2d_1/bias/m/Initializer/zerosConst*
dtype0*
_output_shapes
:`* 
_class
loc:@conv2d_1/bias*
valueB`*    
Х
training/Adam/conv2d_1/bias/mVarHandleOp*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_1/bias/m* 
_class
loc:@conv2d_1/bias*
	container *
shape:`
­
>training/Adam/conv2d_1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/bias/m*
_output_shapes
: * 
_class
loc:@conv2d_1/bias

$training/Adam/conv2d_1/bias/m/AssignAssignVariableOptraining/Adam/conv2d_1/bias/m/training/Adam/conv2d_1/bias/m/Initializer/zeros*
dtype0
­
1training/Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/m* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:`
К
=training/Adam/batch_normalization_1/gamma/m/Initializer/zerosConst*
dtype0*
_output_shapes
:`*.
_class$
" loc:@batch_normalization_1/gamma*
valueB`*    
я
+training/Adam/batch_normalization_1/gamma/mVarHandleOp*
dtype0*
_output_shapes
: *<
shared_name-+training/Adam/batch_normalization_1/gamma/m*.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:`
з
Ltraining/Adam/batch_normalization_1/gamma/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp+training/Adam/batch_normalization_1/gamma/m*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: 
П
2training/Adam/batch_normalization_1/gamma/m/AssignAssignVariableOp+training/Adam/batch_normalization_1/gamma/m=training/Adam/batch_normalization_1/gamma/m/Initializer/zeros*
dtype0
з
?training/Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_1/gamma/m*
dtype0*
_output_shapes
:`*.
_class$
" loc:@batch_normalization_1/gamma
И
<training/Adam/batch_normalization_1/beta/m/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
valueB`*    *
dtype0*
_output_shapes
:`
ь
*training/Adam/batch_normalization_1/beta/mVarHandleOp*
dtype0*
_output_shapes
: *;
shared_name,*training/Adam/batch_normalization_1/beta/m*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:`
д
Ktraining/Adam/batch_normalization_1/beta/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp*training/Adam/batch_normalization_1/beta/m*
_output_shapes
: *-
_class#
!loc:@batch_normalization_1/beta
М
1training/Adam/batch_normalization_1/beta/m/AssignAssignVariableOp*training/Adam/batch_normalization_1/beta/m<training/Adam/batch_normalization_1/beta/m/Initializer/zeros*
dtype0
д
>training/Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_1/beta/m*-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
:`
О
Atraining/Adam/conv2d_2/kernel/m/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_2/kernel*%
valueB"      `      *
dtype0*
_output_shapes
:
 
7training/Adam/conv2d_2/kernel/m/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/conv2d_2/kernel/m/Initializer/zerosFillAtraining/Adam/conv2d_2/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_2/kernel/m/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_2/kernel*

index_type0*'
_output_shapes
:`
и
training/Adam/conv2d_2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
	container *
shape:`
Г
@training/Adam/conv2d_2/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 

&training/Adam/conv2d_2/kernel/m/AssignAssignVariableOptraining/Adam/conv2d_2/kernel/m1training/Adam/conv2d_2/kernel/m/Initializer/zeros*
dtype0
Р
3training/Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
dtype0*'
_output_shapes
:`
 
/training/Adam/conv2d_2/bias/m/Initializer/zerosConst* 
_class
loc:@conv2d_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ц
training/Adam/conv2d_2/bias/mVarHandleOp*
shape:*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
	container 
­
>training/Adam/conv2d_2/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
_output_shapes
: 

$training/Adam/conv2d_2/bias/m/AssignAssignVariableOptraining/Adam/conv2d_2/bias/m/training/Adam/conv2d_2/bias/m/Initializer/zeros*
dtype0
Ў
1training/Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes	
:
М
=training/Adam/batch_normalization_2/gamma/m/Initializer/zerosConst*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*    *
dtype0*
_output_shapes	
:
№
+training/Adam/batch_normalization_2/gamma/mVarHandleOp*
dtype0*
_output_shapes
: *<
shared_name-+training/Adam/batch_normalization_2/gamma/m*.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:
з
Ltraining/Adam/batch_normalization_2/gamma/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp+training/Adam/batch_normalization_2/gamma/m*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: 
П
2training/Adam/batch_normalization_2/gamma/m/AssignAssignVariableOp+training/Adam/batch_normalization_2/gamma/m=training/Adam/batch_normalization_2/gamma/m/Initializer/zeros*
dtype0
и
?training/Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_2/gamma/m*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes	
:
К
<training/Adam/batch_normalization_2/beta/m/Initializer/zerosConst*
dtype0*
_output_shapes	
:*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    
э
*training/Adam/batch_normalization_2/beta/mVarHandleOp*-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:*
dtype0*
_output_shapes
: *;
shared_name,*training/Adam/batch_normalization_2/beta/m
д
Ktraining/Adam/batch_normalization_2/beta/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp*training/Adam/batch_normalization_2/beta/m*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: 
М
1training/Adam/batch_normalization_2/beta/m/AssignAssignVariableOp*training/Adam/batch_normalization_2/beta/m<training/Adam/batch_normalization_2/beta/m/Initializer/zeros*
dtype0
е
>training/Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_2/beta/m*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes	
:
Д
@training/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_1/kernel*
valueB" d     *
dtype0*
_output_shapes
:

6training/Adam/dense_1/kernel/m/Initializer/zeros/ConstConst*!
_class
loc:@dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0training/Adam/dense_1/kernel/m/Initializer/zerosFill@training/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_1/kernel/m/Initializer/zeros/Const*
T0*!
_class
loc:@dense_1/kernel*

index_type0*!
_output_shapes
:Ш 
Я
training/Adam/dense_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: */
shared_name training/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
	container *
shape:Ш 
А
?training/Adam/dense_1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
_output_shapes
: 

%training/Adam/dense_1/kernel/m/AssignAssignVariableOptraining/Adam/dense_1/kernel/m0training/Adam/dense_1/kernel/m/Initializer/zeros*
dtype0
З
2training/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
dtype0*!
_output_shapes
:Ш 
Њ
>training/Adam/dense_1/bias/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@dense_1/bias*
valueB: 

4training/Adam/dense_1/bias/m/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@dense_1/bias*
valueB
 *    

.training/Adam/dense_1/bias/m/Initializer/zerosFill>training/Adam/dense_1/bias/m/Initializer/zeros/shape_as_tensor4training/Adam/dense_1/bias/m/Initializer/zeros/Const*
T0*
_class
loc:@dense_1/bias*

index_type0*
_output_shapes	
: 
У
training/Adam/dense_1/bias/mVarHandleOp*
_class
loc:@dense_1/bias*
	container *
shape: *
dtype0*
_output_shapes
: *-
shared_nametraining/Adam/dense_1/bias/m
Њ
=training/Adam/dense_1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/bias/m*
_output_shapes
: *
_class
loc:@dense_1/bias

#training/Adam/dense_1/bias/m/AssignAssignVariableOptraining/Adam/dense_1/bias/m.training/Adam/dense_1/bias/m/Initializer/zeros*
dtype0
Ћ
0training/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/m*
dtype0*
_output_shapes	
: *
_class
loc:@dense_1/bias
Д
@training/Adam/dense_2/kernel/m/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:

6training/Adam/dense_2/kernel/m/Initializer/zeros/ConstConst*!
_class
loc:@dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0training/Adam/dense_2/kernel/m/Initializer/zerosFill@training/Adam/dense_2/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_2/kernel/m/Initializer/zeros/Const*
T0*!
_class
loc:@dense_2/kernel*

index_type0* 
_output_shapes
:
  
Ю
training/Adam/dense_2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: */
shared_name training/Adam/dense_2/kernel/m*!
_class
loc:@dense_2/kernel*
	container *
shape:
  
А
?training/Adam/dense_2/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_2/kernel/m*!
_class
loc:@dense_2/kernel*
_output_shapes
: 

%training/Adam/dense_2/kernel/m/AssignAssignVariableOptraining/Adam/dense_2/kernel/m0training/Adam/dense_2/kernel/m/Initializer/zeros*
dtype0
Ж
2training/Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/kernel/m*!
_class
loc:@dense_2/kernel*
dtype0* 
_output_shapes
:
  
Њ
>training/Adam/dense_2/bias/m/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense_2/bias*
valueB: *
dtype0*
_output_shapes
:

4training/Adam/dense_2/bias/m/Initializer/zeros/ConstConst*
_class
loc:@dense_2/bias*
valueB
 *    *
dtype0*
_output_shapes
: 

.training/Adam/dense_2/bias/m/Initializer/zerosFill>training/Adam/dense_2/bias/m/Initializer/zeros/shape_as_tensor4training/Adam/dense_2/bias/m/Initializer/zeros/Const*
T0*
_class
loc:@dense_2/bias*

index_type0*
_output_shapes	
: 
У
training/Adam/dense_2/bias/mVarHandleOp*
dtype0*
_output_shapes
: *-
shared_nametraining/Adam/dense_2/bias/m*
_class
loc:@dense_2/bias*
	container *
shape: 
Њ
=training/Adam/dense_2/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_2/bias/m*
_output_shapes
: *
_class
loc:@dense_2/bias

#training/Adam/dense_2/bias/m/AssignAssignVariableOptraining/Adam/dense_2/bias/m.training/Adam/dense_2/bias/m/Initializer/zeros*
dtype0
Ћ
0training/Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/bias/m*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes	
: 
Д
@training/Adam/dense_3/kernel/m/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:

6training/Adam/dense_3/kernel/m/Initializer/zeros/ConstConst*!
_class
loc:@dense_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0training/Adam/dense_3/kernel/m/Initializer/zerosFill@training/Adam/dense_3/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_3/kernel/m/Initializer/zeros/Const*
T0*!
_class
loc:@dense_3/kernel*

index_type0*
_output_shapes
:	 
Э
training/Adam/dense_3/kernel/mVarHandleOp*/
shared_name training/Adam/dense_3/kernel/m*!
_class
loc:@dense_3/kernel*
	container *
shape:	 *
dtype0*
_output_shapes
: 
А
?training/Adam/dense_3/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/kernel/m*!
_class
loc:@dense_3/kernel*
_output_shapes
: 

%training/Adam/dense_3/kernel/m/AssignAssignVariableOptraining/Adam/dense_3/kernel/m0training/Adam/dense_3/kernel/m/Initializer/zeros*
dtype0
Е
2training/Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/kernel/m*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
:	 

.training/Adam/dense_3/bias/m/Initializer/zerosConst*
_class
loc:@dense_3/bias*
valueB*    *
dtype0*
_output_shapes
:
Т
training/Adam/dense_3/bias/mVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *-
shared_nametraining/Adam/dense_3/bias/m*
_class
loc:@dense_3/bias
Њ
=training/Adam/dense_3/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/bias/m*
_class
loc:@dense_3/bias*
_output_shapes
: 

#training/Adam/dense_3/bias/m/AssignAssignVariableOptraining/Adam/dense_3/bias/m.training/Adam/dense_3/bias/m/Initializer/zeros*
dtype0
Њ
0training/Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/bias/m*
dtype0*
_output_shapes
:*
_class
loc:@dense_3/bias
О
Atraining/Adam/conv2d_1/kernel/v/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_1/kernel*%
valueB"         `   
 
7training/Adam/conv2d_1/kernel/v/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/conv2d_1/kernel/v/Initializer/zerosFillAtraining/Adam/conv2d_1/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_1/kernel/v/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_1/kernel*

index_type0*&
_output_shapes
:`
з
training/Adam/conv2d_1/kernel/vVarHandleOp*"
_class
loc:@conv2d_1/kernel*
	container *
shape:`*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_1/kernel/v
Г
@training/Adam/conv2d_1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/kernel/v*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 

&training/Adam/conv2d_1/kernel/v/AssignAssignVariableOptraining/Adam/conv2d_1/kernel/v1training/Adam/conv2d_1/kernel/v/Initializer/zeros*
dtype0
П
3training/Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/v*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:`

/training/Adam/conv2d_1/bias/v/Initializer/zerosConst* 
_class
loc:@conv2d_1/bias*
valueB`*    *
dtype0*
_output_shapes
:`
Х
training/Adam/conv2d_1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
	container *
shape:`
­
>training/Adam/conv2d_1/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 

$training/Adam/conv2d_1/bias/v/AssignAssignVariableOptraining/Adam/conv2d_1/bias/v/training/Adam/conv2d_1/bias/v/Initializer/zeros*
dtype0
­
1training/Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/v*
dtype0*
_output_shapes
:`* 
_class
loc:@conv2d_1/bias
К
=training/Adam/batch_normalization_1/gamma/v/Initializer/zerosConst*.
_class$
" loc:@batch_normalization_1/gamma*
valueB`*    *
dtype0*
_output_shapes
:`
я
+training/Adam/batch_normalization_1/gamma/vVarHandleOp*.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:`*
dtype0*
_output_shapes
: *<
shared_name-+training/Adam/batch_normalization_1/gamma/v
з
Ltraining/Adam/batch_normalization_1/gamma/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp+training/Adam/batch_normalization_1/gamma/v*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: 
П
2training/Adam/batch_normalization_1/gamma/v/AssignAssignVariableOp+training/Adam/batch_normalization_1/gamma/v=training/Adam/batch_normalization_1/gamma/v/Initializer/zeros*
dtype0
з
?training/Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_1/gamma/v*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:`
И
<training/Adam/batch_normalization_1/beta/v/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
valueB`*    *
dtype0*
_output_shapes
:`
ь
*training/Adam/batch_normalization_1/beta/vVarHandleOp*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:`*
dtype0*
_output_shapes
: *;
shared_name,*training/Adam/batch_normalization_1/beta/v
д
Ktraining/Adam/batch_normalization_1/beta/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp*training/Adam/batch_normalization_1/beta/v*
_output_shapes
: *-
_class#
!loc:@batch_normalization_1/beta
М
1training/Adam/batch_normalization_1/beta/v/AssignAssignVariableOp*training/Adam/batch_normalization_1/beta/v<training/Adam/batch_normalization_1/beta/v/Initializer/zeros*
dtype0
д
>training/Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_1/beta/v*
dtype0*
_output_shapes
:`*-
_class#
!loc:@batch_normalization_1/beta
О
Atraining/Adam/conv2d_2/kernel/v/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_2/kernel*%
valueB"      `      *
dtype0*
_output_shapes
:
 
7training/Adam/conv2d_2/kernel/v/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/conv2d_2/kernel/v/Initializer/zerosFillAtraining/Adam/conv2d_2/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_2/kernel/v/Initializer/zeros/Const*'
_output_shapes
:`*
T0*"
_class
loc:@conv2d_2/kernel*

index_type0
и
training/Adam/conv2d_2/kernel/vVarHandleOp*
shape:`*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
	container 
Г
@training/Adam/conv2d_2/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 

&training/Adam/conv2d_2/kernel/v/AssignAssignVariableOptraining/Adam/conv2d_2/kernel/v1training/Adam/conv2d_2/kernel/v/Initializer/zeros*
dtype0
Р
3training/Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/v*
dtype0*'
_output_shapes
:`*"
_class
loc:@conv2d_2/kernel
 
/training/Adam/conv2d_2/bias/v/Initializer/zerosConst* 
_class
loc:@conv2d_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ц
training/Adam/conv2d_2/bias/vVarHandleOp*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
	container *
shape:
­
>training/Adam/conv2d_2/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
_output_shapes
: 

$training/Adam/conv2d_2/bias/v/AssignAssignVariableOptraining/Adam/conv2d_2/bias/v/training/Adam/conv2d_2/bias/v/Initializer/zeros*
dtype0
Ў
1training/Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes	
:
М
=training/Adam/batch_normalization_2/gamma/v/Initializer/zerosConst*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*    *
dtype0*
_output_shapes	
:
№
+training/Adam/batch_normalization_2/gamma/vVarHandleOp*<
shared_name-+training/Adam/batch_normalization_2/gamma/v*.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:*
dtype0*
_output_shapes
: 
з
Ltraining/Adam/batch_normalization_2/gamma/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp+training/Adam/batch_normalization_2/gamma/v*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: 
П
2training/Adam/batch_normalization_2/gamma/v/AssignAssignVariableOp+training/Adam/batch_normalization_2/gamma/v=training/Adam/batch_normalization_2/gamma/v/Initializer/zeros*
dtype0
и
?training/Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp+training/Adam/batch_normalization_2/gamma/v*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes	
:
К
<training/Adam/batch_normalization_2/beta/v/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0*
_output_shapes	
:
э
*training/Adam/batch_normalization_2/beta/vVarHandleOp*-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:*
dtype0*
_output_shapes
: *;
shared_name,*training/Adam/batch_normalization_2/beta/v
д
Ktraining/Adam/batch_normalization_2/beta/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp*training/Adam/batch_normalization_2/beta/v*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: 
М
1training/Adam/batch_normalization_2/beta/v/AssignAssignVariableOp*training/Adam/batch_normalization_2/beta/v<training/Adam/batch_normalization_2/beta/v/Initializer/zeros*
dtype0
е
>training/Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp*training/Adam/batch_normalization_2/beta/v*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes	
:
Д
@training/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_1/kernel*
valueB" d     *
dtype0*
_output_shapes
:

6training/Adam/dense_1/kernel/v/Initializer/zeros/ConstConst*!
_class
loc:@dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0training/Adam/dense_1/kernel/v/Initializer/zerosFill@training/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_1/kernel/v/Initializer/zeros/Const*
T0*!
_class
loc:@dense_1/kernel*

index_type0*!
_output_shapes
:Ш 
Я
training/Adam/dense_1/kernel/vVarHandleOp*
shape:Ш *
dtype0*
_output_shapes
: */
shared_name training/Adam/dense_1/kernel/v*!
_class
loc:@dense_1/kernel*
	container 
А
?training/Adam/dense_1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/kernel/v*
_output_shapes
: *!
_class
loc:@dense_1/kernel

%training/Adam/dense_1/kernel/v/AssignAssignVariableOptraining/Adam/dense_1/kernel/v0training/Adam/dense_1/kernel/v/Initializer/zeros*
dtype0
З
2training/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/v*!
_class
loc:@dense_1/kernel*
dtype0*!
_output_shapes
:Ш 
Њ
>training/Adam/dense_1/bias/v/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense_1/bias*
valueB: *
dtype0*
_output_shapes
:

4training/Adam/dense_1/bias/v/Initializer/zeros/ConstConst*
_class
loc:@dense_1/bias*
valueB
 *    *
dtype0*
_output_shapes
: 

.training/Adam/dense_1/bias/v/Initializer/zerosFill>training/Adam/dense_1/bias/v/Initializer/zeros/shape_as_tensor4training/Adam/dense_1/bias/v/Initializer/zeros/Const*
T0*
_class
loc:@dense_1/bias*

index_type0*
_output_shapes	
: 
У
training/Adam/dense_1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *-
shared_nametraining/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias*
	container *
shape: 
Њ
=training/Adam/dense_1/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/bias/v*
_output_shapes
: *
_class
loc:@dense_1/bias

#training/Adam/dense_1/bias/v/AssignAssignVariableOptraining/Adam/dense_1/bias/v.training/Adam/dense_1/bias/v/Initializer/zeros*
dtype0
Ћ
0training/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes	
: 
Д
@training/Adam/dense_2/kernel/v/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:

6training/Adam/dense_2/kernel/v/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *!
_class
loc:@dense_2/kernel*
valueB
 *    

0training/Adam/dense_2/kernel/v/Initializer/zerosFill@training/Adam/dense_2/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_2/kernel/v/Initializer/zeros/Const* 
_output_shapes
:
  *
T0*!
_class
loc:@dense_2/kernel*

index_type0
Ю
training/Adam/dense_2/kernel/vVarHandleOp*
dtype0*
_output_shapes
: */
shared_name training/Adam/dense_2/kernel/v*!
_class
loc:@dense_2/kernel*
	container *
shape:
  
А
?training/Adam/dense_2/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_2/kernel/v*
_output_shapes
: *!
_class
loc:@dense_2/kernel

%training/Adam/dense_2/kernel/v/AssignAssignVariableOptraining/Adam/dense_2/kernel/v0training/Adam/dense_2/kernel/v/Initializer/zeros*
dtype0
Ж
2training/Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/kernel/v*!
_class
loc:@dense_2/kernel*
dtype0* 
_output_shapes
:
  
Њ
>training/Adam/dense_2/bias/v/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@dense_2/bias*
valueB: 

4training/Adam/dense_2/bias/v/Initializer/zeros/ConstConst*
_class
loc:@dense_2/bias*
valueB
 *    *
dtype0*
_output_shapes
: 

.training/Adam/dense_2/bias/v/Initializer/zerosFill>training/Adam/dense_2/bias/v/Initializer/zeros/shape_as_tensor4training/Adam/dense_2/bias/v/Initializer/zeros/Const*
T0*
_class
loc:@dense_2/bias*

index_type0*
_output_shapes	
: 
У
training/Adam/dense_2/bias/vVarHandleOp*
dtype0*
_output_shapes
: *-
shared_nametraining/Adam/dense_2/bias/v*
_class
loc:@dense_2/bias*
	container *
shape: 
Њ
=training/Adam/dense_2/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_2/bias/v*
_class
loc:@dense_2/bias*
_output_shapes
: 

#training/Adam/dense_2/bias/v/AssignAssignVariableOptraining/Adam/dense_2/bias/v.training/Adam/dense_2/bias/v/Initializer/zeros*
dtype0
Ћ
0training/Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/bias/v*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes	
: 
Д
@training/Adam/dense_3/kernel/v/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*!
_class
loc:@dense_3/kernel*
valueB"      

6training/Adam/dense_3/kernel/v/Initializer/zeros/ConstConst*!
_class
loc:@dense_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0training/Adam/dense_3/kernel/v/Initializer/zerosFill@training/Adam/dense_3/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_3/kernel/v/Initializer/zeros/Const*
T0*!
_class
loc:@dense_3/kernel*

index_type0*
_output_shapes
:	 
Э
training/Adam/dense_3/kernel/vVarHandleOp*
	container *
shape:	 *
dtype0*
_output_shapes
: */
shared_name training/Adam/dense_3/kernel/v*!
_class
loc:@dense_3/kernel
А
?training/Adam/dense_3/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/kernel/v*!
_class
loc:@dense_3/kernel*
_output_shapes
: 

%training/Adam/dense_3/kernel/v/AssignAssignVariableOptraining/Adam/dense_3/kernel/v0training/Adam/dense_3/kernel/v/Initializer/zeros*
dtype0
Е
2training/Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/kernel/v*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
:	 

.training/Adam/dense_3/bias/v/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@dense_3/bias*
valueB*    
Т
training/Adam/dense_3/bias/vVarHandleOp*
_class
loc:@dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes
: *-
shared_nametraining/Adam/dense_3/bias/v
Њ
=training/Adam/dense_3/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/bias/v*
_output_shapes
: *
_class
loc:@dense_3/bias

#training/Adam/dense_3/bias/v/AssignAssignVariableOptraining/Adam/dense_3/bias/v.training/Adam/dense_3/bias/v/Initializer/zeros*
dtype0
Њ
0training/Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/bias/v*
dtype0*
_output_shapes
:*
_class
loc:@dense_3/bias
y
%training/Adam/Identity/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
j
training/Adam/IdentityIdentity%training/Adam/Identity/ReadVariableOp*
T0*
_output_shapes
: 
g
training/Adam/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 
U
training/Adam/add/yConst*
value	B	 R*
dtype0	*
_output_shapes
: 
n
training/Adam/addAddV2training/Adam/ReadVariableOptraining/Adam/add/y*
T0	*
_output_shapes
: 
m
training/Adam/CastCasttraining/Adam/add*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
t
'training/Adam/Identity_1/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
n
training/Adam/Identity_1Identity'training/Adam/Identity_1/ReadVariableOp*
T0*
_output_shapes
: 
t
'training/Adam/Identity_2/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
n
training/Adam/Identity_2Identity'training/Adam/Identity_2/ReadVariableOp*
T0*
_output_shapes
: 
g
training/Adam/PowPowtraining/Adam/Identity_1training/Adam/Cast*
T0*
_output_shapes
: 
i
training/Adam/Pow_1Powtraining/Adam/Identity_2training/Adam/Cast*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
c
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow_1*
_output_shapes
: *
T0
N
training/Adam/SqrtSqrttraining/Adam/sub*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
e
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow*
_output_shapes
: *
T0
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
h
training/Adam/mulMultraining/Adam/Identitytraining/Adam/truediv*
T0*
_output_shapes
: 
X
training/Adam/ConstConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/Identity_1*
T0*
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/Identity_2*
T0*
_output_shapes
: 
Э
;training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdamResourceApplyAdamconv2d_1/kerneltraining/Adam/conv2d_1/kernel/mtraining/Adam/conv2d_1/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstKtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
use_nesterov( 
Л
9training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdamResourceApplyAdamconv2d_1/biastraining/Adam/conv2d_1/bias/mtraining/Adam/conv2d_1/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstCtraining/Adam/gradients/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
use_nesterov( 
Ђ
Gtraining/Adam/Adam/update_batch_normalization_1/gamma/ResourceApplyAdamResourceApplyAdambatch_normalization_1/gamma+training/Adam/batch_normalization_1/gamma/m+training/Adam/batch_normalization_1/gamma/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Constdtraining/Adam/gradients/gradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3:1*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
use_nesterov( 

Ftraining/Adam/Adam/update_batch_normalization_1/beta/ResourceApplyAdamResourceApplyAdambatch_normalization_1/beta*training/Adam/batch_normalization_1/beta/m*training/Adam/batch_normalization_1/beta/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Constdtraining/Adam/gradients/gradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3:2*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
use_nesterov( 
Э
;training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdamResourceApplyAdamconv2d_2/kerneltraining/Adam/conv2d_2/kernel/mtraining/Adam/conv2d_2/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstKtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
use_nesterov( 
Л
9training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdamResourceApplyAdamconv2d_2/biastraining/Adam/conv2d_2/bias/mtraining/Adam/conv2d_2/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstCtraining/Adam/gradients/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0* 
_class
loc:@conv2d_2/bias*
use_nesterov( *
use_locking(
Ђ
Gtraining/Adam/Adam/update_batch_normalization_2/gamma/ResourceApplyAdamResourceApplyAdambatch_normalization_2/gamma+training/Adam/batch_normalization_2/gamma/m+training/Adam/batch_normalization_2/gamma/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Constdtraining/Adam/gradients/gradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3:1*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
use_nesterov( *
use_locking(

Ftraining/Adam/Adam/update_batch_normalization_2/beta/ResourceApplyAdamResourceApplyAdambatch_normalization_2/beta*training/Adam/batch_normalization_2/beta/m*training/Adam/batch_normalization_2/beta/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Constdtraining/Adam/gradients/gradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3:2*
use_nesterov( *
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta
Л
:training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdamResourceApplyAdamdense_1/kerneltraining/Adam/dense_1/kernel/mtraining/Adam/dense_1/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
use_nesterov( 
Е
8training/Adam/Adam/update_dense_1/bias/ResourceApplyAdamResourceApplyAdamdense_1/biastraining/Adam/dense_1/bias/mtraining/Adam/dense_1/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@dense_1/bias*
use_nesterov( 
Л
:training/Adam/Adam/update_dense_2/kernel/ResourceApplyAdamResourceApplyAdamdense_2/kerneltraining/Adam/dense_2/kernel/mtraining/Adam/dense_2/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMul_1*
use_nesterov( *
use_locking(*
T0*!
_class
loc:@dense_2/kernel
Е
8training/Adam/Adam/update_dense_2/bias/ResourceApplyAdamResourceApplyAdamdense_2/biastraining/Adam/dense_2/bias/mtraining/Adam/dense_2/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@dense_2/bias*
use_nesterov( 
Л
:training/Adam/Adam/update_dense_3/kernel/ResourceApplyAdamResourceApplyAdamdense_3/kerneltraining/Adam/dense_3/kernel/mtraining/Adam/dense_3/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMul_1*
use_nesterov( *
use_locking(*
T0*!
_class
loc:@dense_3/kernel
Е
8training/Adam/Adam/update_dense_3/bias/ResourceApplyAdamResourceApplyAdamdense_3/biastraining/Adam/dense_3/bias/mtraining/Adam/dense_3/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
use_nesterov( *
use_locking(*
T0*
_class
loc:@dense_3/bias
м
training/Adam/Adam/ConstConstG^training/Adam/Adam/update_batch_normalization_1/beta/ResourceApplyAdamH^training/Adam/Adam/update_batch_normalization_1/gamma/ResourceApplyAdamG^training/Adam/Adam/update_batch_normalization_2/beta/ResourceApplyAdamH^training/Adam/Adam/update_batch_normalization_2/gamma/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_1/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_2/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_2/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_3/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_3/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: *
value	B	 R
x
&training/Adam/Adam/AssignAddVariableOpAssignAddVariableOptraining/Adam/itertraining/Adam/Adam/Const*
dtype0	

!training/Adam/Adam/ReadVariableOpReadVariableOptraining/Adam/iter'^training/Adam/Adam/AssignAddVariableOpG^training/Adam/Adam/update_batch_normalization_1/beta/ResourceApplyAdamH^training/Adam/Adam/update_batch_normalization_1/gamma/ResourceApplyAdamG^training/Adam/Adam/update_batch_normalization_2/beta/ResourceApplyAdamH^training/Adam/Adam/update_batch_normalization_2/gamma/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_1/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_2/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_2/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_3/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_3/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: 
Q
training_1/group_depsNoOp	^loss/mul'^training/Adam/Adam/AssignAddVariableOp
d
VarIsInitializedOp_18VarIsInitializedOptraining/Adam/conv2d_2/kernel/v*
_output_shapes
: 
d
VarIsInitializedOp_19VarIsInitializedOptraining/Adam/conv2d_1/kernel/m*
_output_shapes
: 
o
VarIsInitializedOp_20VarIsInitializedOp*training/Adam/batch_normalization_1/beta/m*
_output_shapes
: 
d
VarIsInitializedOp_21VarIsInitializedOptraining/Adam/conv2d_2/kernel/m*
_output_shapes
: 
c
VarIsInitializedOp_22VarIsInitializedOptraining/Adam/dense_2/kernel/m*
_output_shapes
: 
c
VarIsInitializedOp_23VarIsInitializedOptraining/Adam/dense_3/kernel/m*
_output_shapes
: 
o
VarIsInitializedOp_24VarIsInitializedOp*training/Adam/batch_normalization_1/beta/v*
_output_shapes
: 
a
VarIsInitializedOp_25VarIsInitializedOptraining/Adam/dense_3/bias/v*
_output_shapes
: 
a
VarIsInitializedOp_26VarIsInitializedOptraining/Adam/dense_1/bias/v*
_output_shapes
: 
c
VarIsInitializedOp_27VarIsInitializedOptraining/Adam/dense_3/kernel/v*
_output_shapes
: 
J
VarIsInitializedOp_28VarIsInitializedOptotal*
_output_shapes
: 
b
VarIsInitializedOp_29VarIsInitializedOptraining/Adam/conv2d_1/bias/m*
_output_shapes
: 
p
VarIsInitializedOp_30VarIsInitializedOp+training/Adam/batch_normalization_1/gamma/m*
_output_shapes
: 
b
VarIsInitializedOp_31VarIsInitializedOptraining/Adam/conv2d_2/bias/m*
_output_shapes
: 
p
VarIsInitializedOp_32VarIsInitializedOp+training/Adam/batch_normalization_1/gamma/v*
_output_shapes
: 
b
VarIsInitializedOp_33VarIsInitializedOptraining/Adam/conv2d_2/bias/v*
_output_shapes
: 
p
VarIsInitializedOp_34VarIsInitializedOp+training/Adam/batch_normalization_2/gamma/m*
_output_shapes
: 
p
VarIsInitializedOp_35VarIsInitializedOp+training/Adam/batch_normalization_2/gamma/v*
_output_shapes
: 
J
VarIsInitializedOp_36VarIsInitializedOpcount*
_output_shapes
: 
Y
VarIsInitializedOp_37VarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 
`
VarIsInitializedOp_38VarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 
a
VarIsInitializedOp_39VarIsInitializedOptraining/Adam/dense_1/bias/m*
_output_shapes
: 
d
VarIsInitializedOp_40VarIsInitializedOptraining/Adam/conv2d_1/kernel/v*
_output_shapes
: 
c
VarIsInitializedOp_41VarIsInitializedOptraining/Adam/dense_2/kernel/v*
_output_shapes
: 
a
VarIsInitializedOp_42VarIsInitializedOptraining/Adam/dense_3/bias/m*
_output_shapes
: 
o
VarIsInitializedOp_43VarIsInitializedOp*training/Adam/batch_normalization_2/beta/v*
_output_shapes
: 
W
VarIsInitializedOp_44VarIsInitializedOptraining/Adam/iter*
_output_shapes
: 
Y
VarIsInitializedOp_45VarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 
X
VarIsInitializedOp_46VarIsInitializedOptraining/Adam/decay*
_output_shapes
: 
o
VarIsInitializedOp_47VarIsInitializedOp*training/Adam/batch_normalization_2/beta/m*
_output_shapes
: 
c
VarIsInitializedOp_48VarIsInitializedOptraining/Adam/dense_1/kernel/m*
_output_shapes
: 
a
VarIsInitializedOp_49VarIsInitializedOptraining/Adam/dense_2/bias/m*
_output_shapes
: 
b
VarIsInitializedOp_50VarIsInitializedOptraining/Adam/conv2d_1/bias/v*
_output_shapes
: 
c
VarIsInitializedOp_51VarIsInitializedOptraining/Adam/dense_1/kernel/v*
_output_shapes
: 
a
VarIsInitializedOp_52VarIsInitializedOptraining/Adam/dense_2/bias/v*
_output_shapes
: 
ў

init_1NoOp^count/Assign^total/Assign2^training/Adam/batch_normalization_1/beta/m/Assign2^training/Adam/batch_normalization_1/beta/v/Assign3^training/Adam/batch_normalization_1/gamma/m/Assign3^training/Adam/batch_normalization_1/gamma/v/Assign2^training/Adam/batch_normalization_2/beta/m/Assign2^training/Adam/batch_normalization_2/beta/v/Assign3^training/Adam/batch_normalization_2/gamma/m/Assign3^training/Adam/batch_normalization_2/gamma/v/Assign^training/Adam/beta_1/Assign^training/Adam/beta_2/Assign%^training/Adam/conv2d_1/bias/m/Assign%^training/Adam/conv2d_1/bias/v/Assign'^training/Adam/conv2d_1/kernel/m/Assign'^training/Adam/conv2d_1/kernel/v/Assign%^training/Adam/conv2d_2/bias/m/Assign%^training/Adam/conv2d_2/bias/v/Assign'^training/Adam/conv2d_2/kernel/m/Assign'^training/Adam/conv2d_2/kernel/v/Assign^training/Adam/decay/Assign$^training/Adam/dense_1/bias/m/Assign$^training/Adam/dense_1/bias/v/Assign&^training/Adam/dense_1/kernel/m/Assign&^training/Adam/dense_1/kernel/v/Assign$^training/Adam/dense_2/bias/m/Assign$^training/Adam/dense_2/bias/v/Assign&^training/Adam/dense_2/kernel/m/Assign&^training/Adam/dense_2/kernel/v/Assign$^training/Adam/dense_3/bias/m/Assign$^training/Adam/dense_3/bias/v/Assign&^training/Adam/dense_3/kernel/m/Assign&^training/Adam/dense_3/kernel/v/Assign^training/Adam/iter/Assign#^training/Adam/learning_rate/Assign
O
Placeholder_18Placeholder*
shape: *
dtype0	*
_output_shapes
: 
X
AssignVariableOp_18AssignVariableOptraining/Adam/iterPlaceholder_18*
dtype0	
r
ReadVariableOp_18ReadVariableOptraining/Adam/iter^AssignVariableOp_18*
dtype0	*
_output_shapes
: 
З
Placeholder_19Placeholder*
dtype0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
e
AssignVariableOp_19AssignVariableOptraining/Adam/conv2d_1/kernel/mPlaceholder_19*
dtype0

ReadVariableOp_19ReadVariableOptraining/Adam/conv2d_1/kernel/m^AssignVariableOp_19*
dtype0*&
_output_shapes
:`
i
Placeholder_20Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
c
AssignVariableOp_20AssignVariableOptraining/Adam/conv2d_1/bias/mPlaceholder_20*
dtype0

ReadVariableOp_20ReadVariableOptraining/Adam/conv2d_1/bias/m^AssignVariableOp_20*
dtype0*
_output_shapes
:`
i
Placeholder_21Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
q
AssignVariableOp_21AssignVariableOp+training/Adam/batch_normalization_1/gamma/mPlaceholder_21*
dtype0

ReadVariableOp_21ReadVariableOp+training/Adam/batch_normalization_1/gamma/m^AssignVariableOp_21*
dtype0*
_output_shapes
:`
i
Placeholder_22Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
p
AssignVariableOp_22AssignVariableOp*training/Adam/batch_normalization_1/beta/mPlaceholder_22*
dtype0

ReadVariableOp_22ReadVariableOp*training/Adam/batch_normalization_1/beta/m^AssignVariableOp_22*
dtype0*
_output_shapes
:`
З
Placeholder_23Placeholder*
dtype0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
e
AssignVariableOp_23AssignVariableOptraining/Adam/conv2d_2/kernel/mPlaceholder_23*
dtype0

ReadVariableOp_23ReadVariableOptraining/Adam/conv2d_2/kernel/m^AssignVariableOp_23*
dtype0*'
_output_shapes
:`
i
Placeholder_24Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
c
AssignVariableOp_24AssignVariableOptraining/Adam/conv2d_2/bias/mPlaceholder_24*
dtype0

ReadVariableOp_24ReadVariableOptraining/Adam/conv2d_2/bias/m^AssignVariableOp_24*
dtype0*
_output_shapes	
:
i
Placeholder_25Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
q
AssignVariableOp_25AssignVariableOp+training/Adam/batch_normalization_2/gamma/mPlaceholder_25*
dtype0

ReadVariableOp_25ReadVariableOp+training/Adam/batch_normalization_2/gamma/m^AssignVariableOp_25*
dtype0*
_output_shapes	
:
i
Placeholder_26Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
p
AssignVariableOp_26AssignVariableOp*training/Adam/batch_normalization_2/beta/mPlaceholder_26*
dtype0

ReadVariableOp_26ReadVariableOp*training/Adam/batch_normalization_2/beta/m^AssignVariableOp_26*
dtype0*
_output_shapes	
:

Placeholder_27Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
d
AssignVariableOp_27AssignVariableOptraining/Adam/dense_1/kernel/mPlaceholder_27*
dtype0

ReadVariableOp_27ReadVariableOptraining/Adam/dense_1/kernel/m^AssignVariableOp_27*
dtype0*!
_output_shapes
:Ш 
i
Placeholder_28Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
b
AssignVariableOp_28AssignVariableOptraining/Adam/dense_1/bias/mPlaceholder_28*
dtype0

ReadVariableOp_28ReadVariableOptraining/Adam/dense_1/bias/m^AssignVariableOp_28*
dtype0*
_output_shapes	
: 

Placeholder_29Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
d
AssignVariableOp_29AssignVariableOptraining/Adam/dense_2/kernel/mPlaceholder_29*
dtype0

ReadVariableOp_29ReadVariableOptraining/Adam/dense_2/kernel/m^AssignVariableOp_29*
dtype0* 
_output_shapes
:
  
i
Placeholder_30Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
b
AssignVariableOp_30AssignVariableOptraining/Adam/dense_2/bias/mPlaceholder_30*
dtype0

ReadVariableOp_30ReadVariableOptraining/Adam/dense_2/bias/m^AssignVariableOp_30*
dtype0*
_output_shapes	
: 

Placeholder_31Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
d
AssignVariableOp_31AssignVariableOptraining/Adam/dense_3/kernel/mPlaceholder_31*
dtype0

ReadVariableOp_31ReadVariableOptraining/Adam/dense_3/kernel/m^AssignVariableOp_31*
dtype0*
_output_shapes
:	 
i
Placeholder_32Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
b
AssignVariableOp_32AssignVariableOptraining/Adam/dense_3/bias/mPlaceholder_32*
dtype0

ReadVariableOp_32ReadVariableOptraining/Adam/dense_3/bias/m^AssignVariableOp_32*
dtype0*
_output_shapes
:
З
Placeholder_33Placeholder*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
e
AssignVariableOp_33AssignVariableOptraining/Adam/conv2d_1/kernel/vPlaceholder_33*
dtype0

ReadVariableOp_33ReadVariableOptraining/Adam/conv2d_1/kernel/v^AssignVariableOp_33*
dtype0*&
_output_shapes
:`
i
Placeholder_34Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
c
AssignVariableOp_34AssignVariableOptraining/Adam/conv2d_1/bias/vPlaceholder_34*
dtype0

ReadVariableOp_34ReadVariableOptraining/Adam/conv2d_1/bias/v^AssignVariableOp_34*
dtype0*
_output_shapes
:`
i
Placeholder_35Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
q
AssignVariableOp_35AssignVariableOp+training/Adam/batch_normalization_1/gamma/vPlaceholder_35*
dtype0

ReadVariableOp_35ReadVariableOp+training/Adam/batch_normalization_1/gamma/v^AssignVariableOp_35*
dtype0*
_output_shapes
:`
i
Placeholder_36Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
p
AssignVariableOp_36AssignVariableOp*training/Adam/batch_normalization_1/beta/vPlaceholder_36*
dtype0

ReadVariableOp_36ReadVariableOp*training/Adam/batch_normalization_1/beta/v^AssignVariableOp_36*
dtype0*
_output_shapes
:`
З
Placeholder_37Placeholder*
dtype0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*?
shape6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
e
AssignVariableOp_37AssignVariableOptraining/Adam/conv2d_2/kernel/vPlaceholder_37*
dtype0

ReadVariableOp_37ReadVariableOptraining/Adam/conv2d_2/kernel/v^AssignVariableOp_37*
dtype0*'
_output_shapes
:`
i
Placeholder_38Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
c
AssignVariableOp_38AssignVariableOptraining/Adam/conv2d_2/bias/vPlaceholder_38*
dtype0

ReadVariableOp_38ReadVariableOptraining/Adam/conv2d_2/bias/v^AssignVariableOp_38*
dtype0*
_output_shapes	
:
i
Placeholder_39Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
q
AssignVariableOp_39AssignVariableOp+training/Adam/batch_normalization_2/gamma/vPlaceholder_39*
dtype0

ReadVariableOp_39ReadVariableOp+training/Adam/batch_normalization_2/gamma/v^AssignVariableOp_39*
dtype0*
_output_shapes	
:
i
Placeholder_40Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
p
AssignVariableOp_40AssignVariableOp*training/Adam/batch_normalization_2/beta/vPlaceholder_40*
dtype0

ReadVariableOp_40ReadVariableOp*training/Adam/batch_normalization_2/beta/v^AssignVariableOp_40*
dtype0*
_output_shapes	
:

Placeholder_41Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
d
AssignVariableOp_41AssignVariableOptraining/Adam/dense_1/kernel/vPlaceholder_41*
dtype0

ReadVariableOp_41ReadVariableOptraining/Adam/dense_1/kernel/v^AssignVariableOp_41*
dtype0*!
_output_shapes
:Ш 
i
Placeholder_42Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
b
AssignVariableOp_42AssignVariableOptraining/Adam/dense_1/bias/vPlaceholder_42*
dtype0

ReadVariableOp_42ReadVariableOptraining/Adam/dense_1/bias/v^AssignVariableOp_42*
dtype0*
_output_shapes	
: 

Placeholder_43Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
d
AssignVariableOp_43AssignVariableOptraining/Adam/dense_2/kernel/vPlaceholder_43*
dtype0

ReadVariableOp_43ReadVariableOptraining/Adam/dense_2/kernel/v^AssignVariableOp_43*
dtype0* 
_output_shapes
:
  
i
Placeholder_44Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
b
AssignVariableOp_44AssignVariableOptraining/Adam/dense_2/bias/vPlaceholder_44*
dtype0

ReadVariableOp_44ReadVariableOptraining/Adam/dense_2/bias/v^AssignVariableOp_44*
dtype0*
_output_shapes	
: 

Placeholder_45Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
d
AssignVariableOp_45AssignVariableOptraining/Adam/dense_3/kernel/vPlaceholder_45*
dtype0

ReadVariableOp_45ReadVariableOptraining/Adam/dense_3/kernel/v^AssignVariableOp_45*
dtype0*
_output_shapes
:	 
i
Placeholder_46Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
b
AssignVariableOp_46AssignVariableOptraining/Adam/dense_3/bias/vPlaceholder_46*
dtype0

ReadVariableOp_46ReadVariableOptraining/Adam/dense_3/bias/v^AssignVariableOp_46*
dtype0*
_output_shapes
:
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_62c7894e20564124a8e1318d0bd6dc77/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ь
save/SaveV2/tensor_namesConst*џ
valueѕBђ3Bbatch_normalization_1/betaBbatch_normalization_1/gammaB!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBbatch_normalization_2/betaBbatch_normalization_2/gammaB!batch_normalization_2/moving_meanB%batch_normalization_2/moving_varianceBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelB*training/Adam/batch_normalization_1/beta/mB*training/Adam/batch_normalization_1/beta/vB+training/Adam/batch_normalization_1/gamma/mB+training/Adam/batch_normalization_1/gamma/vB*training/Adam/batch_normalization_2/beta/mB*training/Adam/batch_normalization_2/beta/vB+training/Adam/batch_normalization_2/gamma/mB+training/Adam/batch_normalization_2/gamma/vBtraining/Adam/beta_1Btraining/Adam/beta_2Btraining/Adam/conv2d_1/bias/mBtraining/Adam/conv2d_1/bias/vBtraining/Adam/conv2d_1/kernel/mBtraining/Adam/conv2d_1/kernel/vBtraining/Adam/conv2d_2/bias/mBtraining/Adam/conv2d_2/bias/vBtraining/Adam/conv2d_2/kernel/mBtraining/Adam/conv2d_2/kernel/vBtraining/Adam/decayBtraining/Adam/dense_1/bias/mBtraining/Adam/dense_1/bias/vBtraining/Adam/dense_1/kernel/mBtraining/Adam/dense_1/kernel/vBtraining/Adam/dense_2/bias/mBtraining/Adam/dense_2/bias/vBtraining/Adam/dense_2/kernel/mBtraining/Adam/dense_2/kernel/vBtraining/Adam/dense_3/bias/mBtraining/Adam/dense_3/bias/vBtraining/Adam/dense_3/kernel/mBtraining/Adam/dense_3/kernel/vBtraining/Adam/iterBtraining/Adam/learning_rate*
dtype0*
_output_shapes
:3
Щ
save/SaveV2/shape_and_slicesConst*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices.batch_normalization_1/beta/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp>training/Adam/batch_normalization_1/beta/m/Read/ReadVariableOp>training/Adam/batch_normalization_1/beta/v/Read/ReadVariableOp?training/Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp?training/Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp>training/Adam/batch_normalization_2/beta/m/Read/ReadVariableOp>training/Adam/batch_normalization_2/beta/v/Read/ReadVariableOp?training/Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp?training/Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp1training/Adam/conv2d_1/bias/m/Read/ReadVariableOp1training/Adam/conv2d_1/bias/v/Read/ReadVariableOp3training/Adam/conv2d_1/kernel/m/Read/ReadVariableOp3training/Adam/conv2d_1/kernel/v/Read/ReadVariableOp1training/Adam/conv2d_2/bias/m/Read/ReadVariableOp1training/Adam/conv2d_2/bias/v/Read/ReadVariableOp3training/Adam/conv2d_2/kernel/m/Read/ReadVariableOp3training/Adam/conv2d_2/kernel/v/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp0training/Adam/dense_1/bias/m/Read/ReadVariableOp0training/Adam/dense_1/bias/v/Read/ReadVariableOp2training/Adam/dense_1/kernel/m/Read/ReadVariableOp2training/Adam/dense_1/kernel/v/Read/ReadVariableOp0training/Adam/dense_2/bias/m/Read/ReadVariableOp0training/Adam/dense_2/bias/v/Read/ReadVariableOp2training/Adam/dense_2/kernel/m/Read/ReadVariableOp2training/Adam/dense_2/kernel/v/Read/ReadVariableOp0training/Adam/dense_3/bias/m/Read/ReadVariableOp0training/Adam/dense_3/bias/v/Read/ReadVariableOp2training/Adam/dense_3/kernel/m/Read/ReadVariableOp2training/Adam/dense_3/kernel/v/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOp*A
dtypes7
523	

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
Я
save/RestoreV2/tensor_namesConst*џ
valueѕBђ3Bbatch_normalization_1/betaBbatch_normalization_1/gammaB!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBbatch_normalization_2/betaBbatch_normalization_2/gammaB!batch_normalization_2/moving_meanB%batch_normalization_2/moving_varianceBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelB*training/Adam/batch_normalization_1/beta/mB*training/Adam/batch_normalization_1/beta/vB+training/Adam/batch_normalization_1/gamma/mB+training/Adam/batch_normalization_1/gamma/vB*training/Adam/batch_normalization_2/beta/mB*training/Adam/batch_normalization_2/beta/vB+training/Adam/batch_normalization_2/gamma/mB+training/Adam/batch_normalization_2/gamma/vBtraining/Adam/beta_1Btraining/Adam/beta_2Btraining/Adam/conv2d_1/bias/mBtraining/Adam/conv2d_1/bias/vBtraining/Adam/conv2d_1/kernel/mBtraining/Adam/conv2d_1/kernel/vBtraining/Adam/conv2d_2/bias/mBtraining/Adam/conv2d_2/bias/vBtraining/Adam/conv2d_2/kernel/mBtraining/Adam/conv2d_2/kernel/vBtraining/Adam/decayBtraining/Adam/dense_1/bias/mBtraining/Adam/dense_1/bias/vBtraining/Adam/dense_1/kernel/mBtraining/Adam/dense_1/kernel/vBtraining/Adam/dense_2/bias/mBtraining/Adam/dense_2/bias/vBtraining/Adam/dense_2/kernel/mBtraining/Adam/dense_2/kernel/vBtraining/Adam/dense_3/bias/mBtraining/Adam/dense_3/bias/vBtraining/Adam/dense_3/kernel/mBtraining/Adam/dense_3/kernel/vBtraining/Adam/iterBtraining/Adam/learning_rate*
dtype0*
_output_shapes
:3
Ь
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:3*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*т
_output_shapesЯ
Ь:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523	
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
c
save/AssignVariableOpAssignVariableOpbatch_normalization_1/betasave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
f
save/AssignVariableOp_1AssignVariableOpbatch_normalization_1/gammasave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
l
save/AssignVariableOp_2AssignVariableOp!batch_normalization_1/moving_meansave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
p
save/AssignVariableOp_3AssignVariableOp%batch_normalization_1/moving_variancesave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
_output_shapes
:*
T0
e
save/AssignVariableOp_4AssignVariableOpbatch_normalization_2/betasave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
f
save/AssignVariableOp_5AssignVariableOpbatch_normalization_2/gammasave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
_output_shapes
:*
T0
l
save/AssignVariableOp_6AssignVariableOp!batch_normalization_2/moving_meansave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
p
save/AssignVariableOp_7AssignVariableOp%batch_normalization_2/moving_variancesave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
X
save/AssignVariableOp_8AssignVariableOpconv2d_1/biassave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
[
save/AssignVariableOp_9AssignVariableOpconv2d_1/kernelsave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
_output_shapes
:*
T0
Z
save/AssignVariableOp_10AssignVariableOpconv2d_2/biassave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
\
save/AssignVariableOp_11AssignVariableOpconv2d_2/kernelsave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
Y
save/AssignVariableOp_12AssignVariableOpdense_1/biassave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
[
save/AssignVariableOp_13AssignVariableOpdense_1/kernelsave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
Y
save/AssignVariableOp_14AssignVariableOpdense_2/biassave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
[
save/AssignVariableOp_15AssignVariableOpdense_2/kernelsave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
Y
save/AssignVariableOp_16AssignVariableOpdense_3/biassave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
[
save/AssignVariableOp_17AssignVariableOpdense_3/kernelsave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
_output_shapes
:*
T0
w
save/AssignVariableOp_18AssignVariableOp*training/Adam/batch_normalization_1/beta/msave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
_output_shapes
:*
T0
w
save/AssignVariableOp_19AssignVariableOp*training/Adam/batch_normalization_1/beta/vsave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
x
save/AssignVariableOp_20AssignVariableOp+training/Adam/batch_normalization_1/gamma/msave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
x
save/AssignVariableOp_21AssignVariableOp+training/Adam/batch_normalization_1/gamma/vsave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
w
save/AssignVariableOp_22AssignVariableOp*training/Adam/batch_normalization_2/beta/msave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
w
save/AssignVariableOp_23AssignVariableOp*training/Adam/batch_normalization_2/beta/vsave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
x
save/AssignVariableOp_24AssignVariableOp+training/Adam/batch_normalization_2/gamma/msave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
x
save/AssignVariableOp_25AssignVariableOp+training/Adam/batch_normalization_2/gamma/vsave/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
_output_shapes
:*
T0
a
save/AssignVariableOp_26AssignVariableOptraining/Adam/beta_1save/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
T0*
_output_shapes
:
a
save/AssignVariableOp_27AssignVariableOptraining/Adam/beta_2save/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
_output_shapes
:*
T0
j
save/AssignVariableOp_28AssignVariableOptraining/Adam/conv2d_1/bias/msave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:
j
save/AssignVariableOp_29AssignVariableOptraining/Adam/conv2d_1/bias/vsave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
l
save/AssignVariableOp_30AssignVariableOptraining/Adam/conv2d_1/kernel/msave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
l
save/AssignVariableOp_31AssignVariableOptraining/Adam/conv2d_1/kernel/vsave/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
j
save/AssignVariableOp_32AssignVariableOptraining/Adam/conv2d_2/bias/msave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
T0*
_output_shapes
:
j
save/AssignVariableOp_33AssignVariableOptraining/Adam/conv2d_2/bias/vsave/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
_output_shapes
:*
T0
l
save/AssignVariableOp_34AssignVariableOptraining/Adam/conv2d_2/kernel/msave/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:
l
save/AssignVariableOp_35AssignVariableOptraining/Adam/conv2d_2/kernel/vsave/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:36*
T0*
_output_shapes
:
`
save/AssignVariableOp_36AssignVariableOptraining/Adam/decaysave/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:37*
T0*
_output_shapes
:
i
save/AssignVariableOp_37AssignVariableOptraining/Adam/dense_1/bias/msave/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:38*
_output_shapes
:*
T0
i
save/AssignVariableOp_38AssignVariableOptraining/Adam/dense_1/bias/vsave/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:39*
T0*
_output_shapes
:
k
save/AssignVariableOp_39AssignVariableOptraining/Adam/dense_1/kernel/msave/Identity_40*
dtype0
R
save/Identity_41Identitysave/RestoreV2:40*
T0*
_output_shapes
:
k
save/AssignVariableOp_40AssignVariableOptraining/Adam/dense_1/kernel/vsave/Identity_41*
dtype0
R
save/Identity_42Identitysave/RestoreV2:41*
T0*
_output_shapes
:
i
save/AssignVariableOp_41AssignVariableOptraining/Adam/dense_2/bias/msave/Identity_42*
dtype0
R
save/Identity_43Identitysave/RestoreV2:42*
T0*
_output_shapes
:
i
save/AssignVariableOp_42AssignVariableOptraining/Adam/dense_2/bias/vsave/Identity_43*
dtype0
R
save/Identity_44Identitysave/RestoreV2:43*
T0*
_output_shapes
:
k
save/AssignVariableOp_43AssignVariableOptraining/Adam/dense_2/kernel/msave/Identity_44*
dtype0
R
save/Identity_45Identitysave/RestoreV2:44*
T0*
_output_shapes
:
k
save/AssignVariableOp_44AssignVariableOptraining/Adam/dense_2/kernel/vsave/Identity_45*
dtype0
R
save/Identity_46Identitysave/RestoreV2:45*
T0*
_output_shapes
:
i
save/AssignVariableOp_45AssignVariableOptraining/Adam/dense_3/bias/msave/Identity_46*
dtype0
R
save/Identity_47Identitysave/RestoreV2:46*
T0*
_output_shapes
:
i
save/AssignVariableOp_46AssignVariableOptraining/Adam/dense_3/bias/vsave/Identity_47*
dtype0
R
save/Identity_48Identitysave/RestoreV2:47*
T0*
_output_shapes
:
k
save/AssignVariableOp_47AssignVariableOptraining/Adam/dense_3/kernel/msave/Identity_48*
dtype0
R
save/Identity_49Identitysave/RestoreV2:48*
_output_shapes
:*
T0
k
save/AssignVariableOp_48AssignVariableOptraining/Adam/dense_3/kernel/vsave/Identity_49*
dtype0
R
save/Identity_50Identitysave/RestoreV2:49*
T0	*
_output_shapes
:
_
save/AssignVariableOp_49AssignVariableOptraining/Adam/itersave/Identity_50*
dtype0	
R
save/Identity_51Identitysave/RestoreV2:50*
T0*
_output_shapes
:
h
save/AssignVariableOp_50AssignVariableOptraining/Adam/learning_ratesave/Identity_51*
dtype0
я

save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_46^save/AssignVariableOp_47^save/AssignVariableOp_48^save/AssignVariableOp_49^save/AssignVariableOp_5^save/AssignVariableOp_50^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"лF
	variablesЭFЪF

conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08
Њ
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
Ї
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
Х
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign7batch_normalization_1/moving_mean/Read/ReadVariableOp:0(25batch_normalization_1/moving_mean/Initializer/zeros:0@H
д
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign;batch_normalization_1/moving_variance/Read/ReadVariableOp:0(28batch_normalization_1/moving_variance/Initializer/ones:0@H

conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08
Њ
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
Ї
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
Х
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign7batch_normalization_2/moving_mean/Read/ReadVariableOp:0(25batch_normalization_2/moving_mean/Initializer/zeros:0@H
д
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign;batch_normalization_2/moving_variance/Read/ReadVariableOp:0(28batch_normalization_2/moving_variance/Initializer/ones:0@H

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08

dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08

dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08

training/Adam/iter:0training/Adam/iter/Assign(training/Adam/iter/Read/ReadVariableOp:0(2&training/Adam/iter/Initializer/zeros:0H

training/Adam/beta_1:0training/Adam/beta_1/Assign*training/Adam/beta_1/Read/ReadVariableOp:0(20training/Adam/beta_1/Initializer/initial_value:0H

training/Adam/beta_2:0training/Adam/beta_2/Assign*training/Adam/beta_2/Read/ReadVariableOp:0(20training/Adam/beta_2/Initializer/initial_value:0H

training/Adam/decay:0training/Adam/decay/Assign)training/Adam/decay/Read/ReadVariableOp:0(2/training/Adam/decay/Initializer/initial_value:0H
Г
training/Adam/learning_rate:0"training/Adam/learning_rate/Assign1training/Adam/learning_rate/Read/ReadVariableOp:0(27training/Adam/learning_rate/Initializer/initial_value:0H
Й
!training/Adam/conv2d_1/kernel/m:0&training/Adam/conv2d_1/kernel/m/Assign5training/Adam/conv2d_1/kernel/m/Read/ReadVariableOp:0(23training/Adam/conv2d_1/kernel/m/Initializer/zeros:0
Б
training/Adam/conv2d_1/bias/m:0$training/Adam/conv2d_1/bias/m/Assign3training/Adam/conv2d_1/bias/m/Read/ReadVariableOp:0(21training/Adam/conv2d_1/bias/m/Initializer/zeros:0
щ
-training/Adam/batch_normalization_1/gamma/m:02training/Adam/batch_normalization_1/gamma/m/AssignAtraining/Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp:0(2?training/Adam/batch_normalization_1/gamma/m/Initializer/zeros:0
х
,training/Adam/batch_normalization_1/beta/m:01training/Adam/batch_normalization_1/beta/m/Assign@training/Adam/batch_normalization_1/beta/m/Read/ReadVariableOp:0(2>training/Adam/batch_normalization_1/beta/m/Initializer/zeros:0
Й
!training/Adam/conv2d_2/kernel/m:0&training/Adam/conv2d_2/kernel/m/Assign5training/Adam/conv2d_2/kernel/m/Read/ReadVariableOp:0(23training/Adam/conv2d_2/kernel/m/Initializer/zeros:0
Б
training/Adam/conv2d_2/bias/m:0$training/Adam/conv2d_2/bias/m/Assign3training/Adam/conv2d_2/bias/m/Read/ReadVariableOp:0(21training/Adam/conv2d_2/bias/m/Initializer/zeros:0
щ
-training/Adam/batch_normalization_2/gamma/m:02training/Adam/batch_normalization_2/gamma/m/AssignAtraining/Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp:0(2?training/Adam/batch_normalization_2/gamma/m/Initializer/zeros:0
х
,training/Adam/batch_normalization_2/beta/m:01training/Adam/batch_normalization_2/beta/m/Assign@training/Adam/batch_normalization_2/beta/m/Read/ReadVariableOp:0(2>training/Adam/batch_normalization_2/beta/m/Initializer/zeros:0
Е
 training/Adam/dense_1/kernel/m:0%training/Adam/dense_1/kernel/m/Assign4training/Adam/dense_1/kernel/m/Read/ReadVariableOp:0(22training/Adam/dense_1/kernel/m/Initializer/zeros:0
­
training/Adam/dense_1/bias/m:0#training/Adam/dense_1/bias/m/Assign2training/Adam/dense_1/bias/m/Read/ReadVariableOp:0(20training/Adam/dense_1/bias/m/Initializer/zeros:0
Е
 training/Adam/dense_2/kernel/m:0%training/Adam/dense_2/kernel/m/Assign4training/Adam/dense_2/kernel/m/Read/ReadVariableOp:0(22training/Adam/dense_2/kernel/m/Initializer/zeros:0
­
training/Adam/dense_2/bias/m:0#training/Adam/dense_2/bias/m/Assign2training/Adam/dense_2/bias/m/Read/ReadVariableOp:0(20training/Adam/dense_2/bias/m/Initializer/zeros:0
Е
 training/Adam/dense_3/kernel/m:0%training/Adam/dense_3/kernel/m/Assign4training/Adam/dense_3/kernel/m/Read/ReadVariableOp:0(22training/Adam/dense_3/kernel/m/Initializer/zeros:0
­
training/Adam/dense_3/bias/m:0#training/Adam/dense_3/bias/m/Assign2training/Adam/dense_3/bias/m/Read/ReadVariableOp:0(20training/Adam/dense_3/bias/m/Initializer/zeros:0
Й
!training/Adam/conv2d_1/kernel/v:0&training/Adam/conv2d_1/kernel/v/Assign5training/Adam/conv2d_1/kernel/v/Read/ReadVariableOp:0(23training/Adam/conv2d_1/kernel/v/Initializer/zeros:0
Б
training/Adam/conv2d_1/bias/v:0$training/Adam/conv2d_1/bias/v/Assign3training/Adam/conv2d_1/bias/v/Read/ReadVariableOp:0(21training/Adam/conv2d_1/bias/v/Initializer/zeros:0
щ
-training/Adam/batch_normalization_1/gamma/v:02training/Adam/batch_normalization_1/gamma/v/AssignAtraining/Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp:0(2?training/Adam/batch_normalization_1/gamma/v/Initializer/zeros:0
х
,training/Adam/batch_normalization_1/beta/v:01training/Adam/batch_normalization_1/beta/v/Assign@training/Adam/batch_normalization_1/beta/v/Read/ReadVariableOp:0(2>training/Adam/batch_normalization_1/beta/v/Initializer/zeros:0
Й
!training/Adam/conv2d_2/kernel/v:0&training/Adam/conv2d_2/kernel/v/Assign5training/Adam/conv2d_2/kernel/v/Read/ReadVariableOp:0(23training/Adam/conv2d_2/kernel/v/Initializer/zeros:0
Б
training/Adam/conv2d_2/bias/v:0$training/Adam/conv2d_2/bias/v/Assign3training/Adam/conv2d_2/bias/v/Read/ReadVariableOp:0(21training/Adam/conv2d_2/bias/v/Initializer/zeros:0
щ
-training/Adam/batch_normalization_2/gamma/v:02training/Adam/batch_normalization_2/gamma/v/AssignAtraining/Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp:0(2?training/Adam/batch_normalization_2/gamma/v/Initializer/zeros:0
х
,training/Adam/batch_normalization_2/beta/v:01training/Adam/batch_normalization_2/beta/v/Assign@training/Adam/batch_normalization_2/beta/v/Read/ReadVariableOp:0(2>training/Adam/batch_normalization_2/beta/v/Initializer/zeros:0
Е
 training/Adam/dense_1/kernel/v:0%training/Adam/dense_1/kernel/v/Assign4training/Adam/dense_1/kernel/v/Read/ReadVariableOp:0(22training/Adam/dense_1/kernel/v/Initializer/zeros:0
­
training/Adam/dense_1/bias/v:0#training/Adam/dense_1/bias/v/Assign2training/Adam/dense_1/bias/v/Read/ReadVariableOp:0(20training/Adam/dense_1/bias/v/Initializer/zeros:0
Е
 training/Adam/dense_2/kernel/v:0%training/Adam/dense_2/kernel/v/Assign4training/Adam/dense_2/kernel/v/Read/ReadVariableOp:0(22training/Adam/dense_2/kernel/v/Initializer/zeros:0
­
training/Adam/dense_2/bias/v:0#training/Adam/dense_2/bias/v/Assign2training/Adam/dense_2/bias/v/Read/ReadVariableOp:0(20training/Adam/dense_2/bias/v/Initializer/zeros:0
Е
 training/Adam/dense_3/kernel/v:0%training/Adam/dense_3/kernel/v/Assign4training/Adam/dense_3/kernel/v/Read/ReadVariableOp:0(22training/Adam/dense_3/kernel/v/Initializer/zeros:0
­
training/Adam/dense_3/bias/v:0#training/Adam/dense_3/bias/v/Assign2training/Adam/dense_3/bias/v/Read/ReadVariableOp:0(20training/Adam/dense_3/bias/v/Initializer/zeros:0"
trainable_variables

conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08
Њ
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
Ї
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08

conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08
Њ
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
Ї
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08

dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08

dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08*Б
serving_default
@
input_image1
conv2d_1_input:0џџџџџџџџџрр=
dense_3/Softmax:0(
dense_3/Softmax:0џџџџџџџџџtensorflow/serving/predict