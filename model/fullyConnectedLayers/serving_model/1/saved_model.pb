аб
#ъ"
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
Ttype"serve*1.15.02v1.15.0-rc3-22-g590d6eef7eђі
~
input_1Placeholder*&
shape:џџџџџџџџџрр*
dtype0*1
_output_shapes
:џџџџџџџџџрр
V
flatten_1/ShapeShapeinput_1*
_output_shapes
:*
T0*
out_type0
g
flatten_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
i
flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
i
flatten_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ћ
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
shrink_axis_mask*
end_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask *
Index0*
T0*
new_axis_mask 
d
flatten_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ

flatten_1/Reshape/shapePackflatten_1/strided_sliceflatten_1/Reshape/shape/1*
T0*
N*

axis *
_output_shapes
:

flatten_1/ReshapeReshapeinput_1flatten_1/Reshape/shape*
Tshape0*)
_output_shapes
:џџџџџџџџџ	*
T0
Ѓ
/dense_1/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
valueB" L    *
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *шЬЛ*!
_class
loc:@dense_1/kernel

-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *шЬ;*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
ю
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*!
_output_shapes
:	 *
T0*!
_class
loc:@dense_1/kernel*
dtype0*
seed2 *

seed 
ж
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_1/kernel
ы
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_output_shapes
:	 *
T0*!
_class
loc:@dense_1/kernel
н
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:	 
Џ
dense_1/kernelVarHandleOp*
_output_shapes
: *
shape:	 *
shared_namedense_1/kernel*
	container *!
_class
loc:@dense_1/kernel*
dtype0
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
:	 

.dense_1/bias/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: *
_class
loc:@dense_1/bias

$dense_1/bias/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
_class
loc:@dense_1/bias*
dtype0
е
dense_1/bias/Initializer/zerosFill.dense_1/bias/Initializer/zeros/shape_as_tensor$dense_1/bias/Initializer/zeros/Const*
T0*
_output_shapes	
: *

index_type0*
_class
loc:@dense_1/bias
Ѓ
dense_1/biasVarHandleOp*
dtype0*
shared_namedense_1/bias*
shape: *
_class
loc:@dense_1/bias*
_output_shapes
: *
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
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*!
_output_shapes
:	 *
dtype0
Ѓ
dense_1/MatMulMatMulflatten_1/Reshapedense_1/MatMul/ReadVariableOp*(
_output_shapes
:џџџџџџџџџ *
transpose_b( *
transpose_a( *
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
dense_1/ReluReludense_1/BiasAdd*(
_output_shapes
:џџџџџџџџџ *
T0
_
dropout_1/IdentityIdentitydense_1/Relu*
T0*(
_output_shapes
:џџџџџџџџџ 
Ѓ
/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*!
_class
loc:@dense_2/kernel*
dtype0*
valueB"      

-dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *зГнМ*
_output_shapes
: *!
_class
loc:@dense_2/kernel

-dense_2/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_2/kernel*
valueB
 *зГн<*
_output_shapes
: *
dtype0
э
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*

seed *!
_class
loc:@dense_2/kernel*
T0* 
_output_shapes
:
  *
seed2 *
dtype0
ж
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
T0*
_output_shapes
: 
ъ
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
  *
T0*!
_class
loc:@dense_2/kernel
м
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
  *!
_class
loc:@dense_2/kernel
Ў
dense_2/kernelVarHandleOp*!
_class
loc:@dense_2/kernel*
dtype0*
shared_namedense_2/kernel*
_output_shapes
: *
shape:
  *
	container 
m
/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/kernel*
_output_shapes
: 
q
dense_2/kernel/AssignAssignVariableOpdense_2/kernel)dense_2/kernel/Initializer/random_uniform*
dtype0
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
  *
dtype0
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
_class
loc:@dense_2/bias*
_output_shapes
: *
valueB
 *    *
dtype0
е
dense_2/bias/Initializer/zerosFill.dense_2/bias/Initializer/zeros/shape_as_tensor$dense_2/bias/Initializer/zeros/Const*

index_type0*
_output_shapes	
: *
_class
loc:@dense_2/bias*
T0
Ѓ
dense_2/biasVarHandleOp*
dtype0*
_class
loc:@dense_2/bias*
	container *
shared_namedense_2/bias*
shape: *
_output_shapes
: 
i
-dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/bias*
_output_shapes
: 
b
dense_2/bias/AssignAssignVariableOpdense_2/biasdense_2/bias/Initializer/zeros*
dtype0
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
: *
dtype0
n
dense_2/MatMul/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
  *
dtype0
Є
dense_2/MatMulMatMuldropout_1/Identitydense_2/MatMul/ReadVariableOp*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ *
transpose_a( 
h
dense_2/BiasAdd/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
: 

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ *
data_formatNHWC
X
dense_2/ReluReludense_2/BiasAdd*(
_output_shapes
:џџџџџџџџџ *
T0
Ѓ
/dense_3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"      *!
_class
loc:@dense_3/kernel

-dense_3/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГнМ

-dense_3/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_3/kernel*
_output_shapes
: *
valueB
 *зГн<*
dtype0
э
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
seed2 *

seed *
dtype0*
T0* 
_output_shapes
:
  *!
_class
loc:@dense_3/kernel
ж
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
T0
ъ
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_3/kernel* 
_output_shapes
:
  *
T0
м
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min* 
_output_shapes
:
  *
T0*!
_class
loc:@dense_3/kernel
Ў
dense_3/kernelVarHandleOp*
shape:
  *
_output_shapes
: *
	container *!
_class
loc:@dense_3/kernel*
shared_namedense_3/kernel*
dtype0
m
/dense_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/kernel*
_output_shapes
: 
q
dense_3/kernel/AssignAssignVariableOpdense_3/kernel)dense_3/kernel/Initializer/random_uniform*
dtype0
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
  *
dtype0

.dense_3/bias/Initializer/zeros/shape_as_tensorConst*
valueB: *
_output_shapes
:*
dtype0*
_class
loc:@dense_3/bias

$dense_3/bias/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@dense_3/bias
е
dense_3/bias/Initializer/zerosFill.dense_3/bias/Initializer/zeros/shape_as_tensor$dense_3/bias/Initializer/zeros/Const*

index_type0*
_class
loc:@dense_3/bias*
_output_shapes	
: *
T0
Ѓ
dense_3/biasVarHandleOp*
_output_shapes
: *
	container *
dtype0*
shared_namedense_3/bias*
_class
loc:@dense_3/bias*
shape: 
i
-dense_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/bias*
_output_shapes
: 
b
dense_3/bias/AssignAssignVariableOpdense_3/biasdense_3/bias/Initializer/zeros*
dtype0
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
: *
dtype0
n
dense_3/MatMul/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0* 
_output_shapes
:
  

dense_3/MatMulMatMuldense_2/Reludense_3/MatMul/ReadVariableOp*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:џџџџџџџџџ 
h
dense_3/BiasAdd/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
: *
dtype0

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/BiasAdd/ReadVariableOp*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ *
T0
X
dense_3/ReluReludense_3/BiasAdd*(
_output_shapes
:џџџџџџџџџ *
T0
_
dropout_2/IdentityIdentitydense_3/Relu*
T0*(
_output_shapes
:џџџџџџџџџ 
Ѓ
/dense_4/kernel/Initializer/random_uniform/shapeConst*
valueB"      *!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
:

-dense_4/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *зГнМ*!
_class
loc:@dense_4/kernel*
_output_shapes
: 

-dense_4/kernel/Initializer/random_uniform/maxConst*
valueB
 *зГн<*
_output_shapes
: *
dtype0*!
_class
loc:@dense_4/kernel
э
7dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_4/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
  *!
_class
loc:@dense_4/kernel*

seed *
seed2 *
dtype0*
T0
ж
-dense_4/kernel/Initializer/random_uniform/subSub-dense_4/kernel/Initializer/random_uniform/max-dense_4/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_4/kernel*
_output_shapes
: *
T0
ъ
-dense_4/kernel/Initializer/random_uniform/mulMul7dense_4/kernel/Initializer/random_uniform/RandomUniform-dense_4/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_4/kernel*
T0* 
_output_shapes
:
  
м
)dense_4/kernel/Initializer/random_uniformAdd-dense_4/kernel/Initializer/random_uniform/mul-dense_4/kernel/Initializer/random_uniform/min* 
_output_shapes
:
  *
T0*!
_class
loc:@dense_4/kernel
Ў
dense_4/kernelVarHandleOp*
	container *
shared_namedense_4/kernel*
shape:
  *
dtype0*!
_class
loc:@dense_4/kernel*
_output_shapes
: 
m
/dense_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/kernel*
_output_shapes
: 
q
dense_4/kernel/AssignAssignVariableOpdense_4/kernel)dense_4/kernel/Initializer/random_uniform*
dtype0
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
  *
dtype0

.dense_4/bias/Initializer/zeros/shape_as_tensorConst*
valueB: *
_class
loc:@dense_4/bias*
_output_shapes
:*
dtype0

$dense_4/bias/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *
_class
loc:@dense_4/bias*
_output_shapes
: 
е
dense_4/bias/Initializer/zerosFill.dense_4/bias/Initializer/zeros/shape_as_tensor$dense_4/bias/Initializer/zeros/Const*
T0*
_output_shapes	
: *

index_type0*
_class
loc:@dense_4/bias
Ѓ
dense_4/biasVarHandleOp*
dtype0*
_class
loc:@dense_4/bias*
shared_namedense_4/bias*
shape: *
	container *
_output_shapes
: 
i
-dense_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/bias*
_output_shapes
: 
b
dense_4/bias/AssignAssignVariableOpdense_4/biasdense_4/bias/Initializer/zeros*
dtype0
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes	
: 
n
dense_4/MatMul/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
  *
dtype0
Є
dense_4/MatMulMatMuldropout_2/Identitydense_4/MatMul/ReadVariableOp*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:џџџџџџџџџ 
h
dense_4/BiasAdd/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes	
: 

dense_4/BiasAddBiasAdddense_4/MatMuldense_4/BiasAdd/ReadVariableOp*(
_output_shapes
:џџџџџџџџџ *
data_formatNHWC*
T0
X
dense_4/ReluReludense_4/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ 
Ѓ
/dense_5/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *!
_class
loc:@dense_5/kernel*
dtype0

-dense_5/kernel/Initializer/random_uniform/minConst*
valueB
 *IvН*!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes
: 

-dense_5/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *Iv=*!
_class
loc:@dense_5/kernel
ь
7dense_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_5/kernel/Initializer/random_uniform/shape*
seed2 *

seed *
_output_shapes
:	 *!
_class
loc:@dense_5/kernel*
T0*
dtype0
ж
-dense_5/kernel/Initializer/random_uniform/subSub-dense_5/kernel/Initializer/random_uniform/max-dense_5/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_5/kernel
щ
-dense_5/kernel/Initializer/random_uniform/mulMul7dense_5/kernel/Initializer/random_uniform/RandomUniform-dense_5/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_5/kernel*
_output_shapes
:	 *
T0
л
)dense_5/kernel/Initializer/random_uniformAdd-dense_5/kernel/Initializer/random_uniform/mul-dense_5/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_5/kernel*
T0*
_output_shapes
:	 
­
dense_5/kernelVarHandleOp*
dtype0*
shape:	 *
_output_shapes
: *
shared_namedense_5/kernel*
	container *!
_class
loc:@dense_5/kernel
m
/dense_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/kernel*
_output_shapes
: 
q
dense_5/kernel/AssignAssignVariableOpdense_5/kernel)dense_5/kernel/Initializer/random_uniform*
dtype0
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
dtype0*
_output_shapes
:	 

dense_5/bias/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *
_class
loc:@dense_5/bias
Ђ
dense_5/biasVarHandleOp*
	container *
shared_namedense_5/bias*
_class
loc:@dense_5/bias*
shape:*
_output_shapes
: *
dtype0
i
-dense_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/bias*
_output_shapes
: 
b
dense_5/bias/AssignAssignVariableOpdense_5/biasdense_5/bias/Initializer/zeros*
dtype0
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
m
dense_5/MatMul/ReadVariableOpReadVariableOpdense_5/kernel*
dtype0*
_output_shapes
:	 

dense_5/MatMulMatMuldense_4/Reludense_5/MatMul/ReadVariableOp*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
g
dense_5/BiasAdd/ReadVariableOpReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:

dense_5/BiasAddBiasAdddense_5/MatMuldense_5/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
]
dense_5/SoftmaxSoftmaxdense_5/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ

PlaceholderPlaceholder*%
shape:џџџџџџџџџџџџџџџџџџ*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
N
AssignVariableOpAssignVariableOpdense_1/kernelPlaceholder*
dtype0
s
ReadVariableOpReadVariableOpdense_1/kernel^AssignVariableOp*
dtype0*!
_output_shapes
:	 
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
P
AssignVariableOp_1AssignVariableOpdense_1/biasPlaceholder_1*
dtype0
o
ReadVariableOp_1ReadVariableOpdense_1/bias^AssignVariableOp_1*
_output_shapes	
: *
dtype0

Placeholder_2Placeholder*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ*
dtype0
R
AssignVariableOp_2AssignVariableOpdense_2/kernelPlaceholder_2*
dtype0
v
ReadVariableOp_2ReadVariableOpdense_2/kernel^AssignVariableOp_2* 
_output_shapes
:
  *
dtype0
h
Placeholder_3Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
P
AssignVariableOp_3AssignVariableOpdense_2/biasPlaceholder_3*
dtype0
o
ReadVariableOp_3ReadVariableOpdense_2/bias^AssignVariableOp_3*
_output_shapes	
: *
dtype0

Placeholder_4Placeholder*
dtype0*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
R
AssignVariableOp_4AssignVariableOpdense_3/kernelPlaceholder_4*
dtype0
v
ReadVariableOp_4ReadVariableOpdense_3/kernel^AssignVariableOp_4*
dtype0* 
_output_shapes
:
  
h
Placeholder_5Placeholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
P
AssignVariableOp_5AssignVariableOpdense_3/biasPlaceholder_5*
dtype0
o
ReadVariableOp_5ReadVariableOpdense_3/bias^AssignVariableOp_5*
dtype0*
_output_shapes	
: 

Placeholder_6Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
R
AssignVariableOp_6AssignVariableOpdense_4/kernelPlaceholder_6*
dtype0
v
ReadVariableOp_6ReadVariableOpdense_4/kernel^AssignVariableOp_6* 
_output_shapes
:
  *
dtype0
h
Placeholder_7Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
P
AssignVariableOp_7AssignVariableOpdense_4/biasPlaceholder_7*
dtype0
o
ReadVariableOp_7ReadVariableOpdense_4/bias^AssignVariableOp_7*
dtype0*
_output_shapes	
: 

Placeholder_8Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
R
AssignVariableOp_8AssignVariableOpdense_5/kernelPlaceholder_8*
dtype0
u
ReadVariableOp_8ReadVariableOpdense_5/kernel^AssignVariableOp_8*
_output_shapes
:	 *
dtype0
h
Placeholder_9Placeholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
P
AssignVariableOp_9AssignVariableOpdense_5/biasPlaceholder_9*
dtype0
n
ReadVariableOp_9ReadVariableOpdense_5/bias^AssignVariableOp_9*
_output_shapes
:*
dtype0
P
VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
P
VarIsInitializedOp_1VarIsInitializedOpdense_1/bias*
_output_shapes
: 
R
VarIsInitializedOp_2VarIsInitializedOpdense_2/kernel*
_output_shapes
: 
R
VarIsInitializedOp_3VarIsInitializedOpdense_3/kernel*
_output_shapes
: 
P
VarIsInitializedOp_4VarIsInitializedOpdense_2/bias*
_output_shapes
: 
P
VarIsInitializedOp_5VarIsInitializedOpdense_3/bias*
_output_shapes
: 
R
VarIsInitializedOp_6VarIsInitializedOpdense_4/kernel*
_output_shapes
: 
P
VarIsInitializedOp_7VarIsInitializedOpdense_4/bias*
_output_shapes
: 
R
VarIsInitializedOp_8VarIsInitializedOpdense_5/kernel*
_output_shapes
: 
P
VarIsInitializedOp_9VarIsInitializedOpdense_5/bias*
_output_shapes
: 
ђ
initNoOp^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^dense_5/bias/Assign^dense_5/kernel/Assign

dense_5_targetPlaceholder*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*%
shape:џџџџџџџџџџџџџџџџџџ
v
total/Initializer/zerosConst*
_output_shapes
: *
_class

loc:@total*
valueB
 *    *
dtype0

totalVarHandleOp*
shape: *
shared_nametotal*
	container *
dtype0*
_output_shapes
: *
_class

loc:@total
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
count/Initializer/zerosConst*
_class

loc:@count*
valueB
 *    *
_output_shapes
: *
dtype0

countVarHandleOp*
_class

loc:@count*
dtype0*
	container *
shared_namecount*
_output_shapes
: *
shape: 
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
metrics/acc/ArgMax/dimensionConst*
dtype0*
valueB :
џџџџџџџџџ*
_output_shapes
: 

metrics/acc/ArgMaxArgMaxdense_5_targetmetrics/acc/ArgMax/dimension*
T0*#
_output_shapes
:џџџџџџџџџ*
output_type0	*

Tidx0
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

metrics/acc/ArgMax_1ArgMaxdense_5/Softmaxmetrics/acc/ArgMax_1/dimension*#
_output_shapes
:џџџџџџџџџ*
output_type0	*
T0*

Tidx0

metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
incompatible_shape_error(*
T0	*#
_output_shapes
:џџџџџџџџџ
x
metrics/acc/CastCastmetrics/acc/Equal*

DstT0*#
_output_shapes
:џџџџџџџџџ*

SrcT0
*
Truncate( 
[
metrics/acc/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
y
metrics/acc/SumSummetrics/acc/Castmetrics/acc/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0

metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
_output_shapes
: *
dtype0
[
metrics/acc/SizeSizemetrics/acc/Cast*
T0*
out_type0*
_output_shapes
: 
l
metrics/acc/Cast_1Castmetrics/acc/Size*

SrcT0*

DstT0*
Truncate( *
_output_shapes
: 
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
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
_output_shapes
: *
dtype0

metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
_output_shapes
: *
T0
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
_output_shapes
: *
T0
\
loss/dense_5_loss/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
8loss/dense_5_loss/softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 

9loss/dense_5_loss/softmax_cross_entropy_with_logits/ShapeShapedense_5/BiasAdd*
_output_shapes
:*
T0*
out_type0
|
:loss/dense_5_loss/softmax_cross_entropy_with_logits/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :

;loss/dense_5_loss/softmax_cross_entropy_with_logits/Shape_1Shapedense_5/BiasAdd*
out_type0*
_output_shapes
:*
T0
{
9loss/dense_5_loss/softmax_cross_entropy_with_logits/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
ж
7loss/dense_5_loss/softmax_cross_entropy_with_logits/SubSub:loss/dense_5_loss/softmax_cross_entropy_with_logits/Rank_19loss/dense_5_loss/softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
К
?loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice/beginPack7loss/dense_5_loss/softmax_cross_entropy_with_logits/Sub*

axis *
N*
T0*
_output_shapes
:

>loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
В
9loss/dense_5_loss/softmax_cross_entropy_with_logits/SliceSlice;loss/dense_5_loss/softmax_cross_entropy_with_logits/Shape_1?loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice/begin>loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice/size*
Index0*
_output_shapes
:*
T0

Closs/dense_5_loss/softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:

?loss/dense_5_loss/softmax_cross_entropy_with_logits/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
С
:loss/dense_5_loss/softmax_cross_entropy_with_logits/concatConcatV2Closs/dense_5_loss/softmax_cross_entropy_with_logits/concat/values_09loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice?loss/dense_5_loss/softmax_cross_entropy_with_logits/concat/axis*

Tidx0*
_output_shapes
:*
N*
T0
м
;loss/dense_5_loss/softmax_cross_entropy_with_logits/ReshapeReshapedense_5/BiasAdd:loss/dense_5_loss/softmax_cross_entropy_with_logits/concat*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Tshape0*
T0
|
:loss/dense_5_loss/softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
_output_shapes
: *
dtype0

;loss/dense_5_loss/softmax_cross_entropy_with_logits/Shape_2Shapedense_5_target*
T0*
_output_shapes
:*
out_type0
}
;loss/dense_5_loss/softmax_cross_entropy_with_logits/Sub_1/yConst*
dtype0*
value	B :*
_output_shapes
: 
к
9loss/dense_5_loss/softmax_cross_entropy_with_logits/Sub_1Sub:loss/dense_5_loss/softmax_cross_entropy_with_logits/Rank_2;loss/dense_5_loss/softmax_cross_entropy_with_logits/Sub_1/y*
_output_shapes
: *
T0
О
Aloss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_1/beginPack9loss/dense_5_loss/softmax_cross_entropy_with_logits/Sub_1*
N*
_output_shapes
:*

axis *
T0

@loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
И
;loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_1Slice;loss/dense_5_loss/softmax_cross_entropy_with_logits/Shape_2Aloss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_1/begin@loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:

Eloss/dense_5_loss/softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
џџџџџџџџџ*
_output_shapes
:*
dtype0

Aloss/dense_5_loss/softmax_cross_entropy_with_logits/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Щ
<loss/dense_5_loss/softmax_cross_entropy_with_logits/concat_1ConcatV2Eloss/dense_5_loss/softmax_cross_entropy_with_logits/concat_1/values_0;loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_1Aloss/dense_5_loss/softmax_cross_entropy_with_logits/concat_1/axis*
_output_shapes
:*
T0*
N*

Tidx0
п
=loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_1Reshapedense_5_target<loss/dense_5_loss/softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

3loss/dense_5_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits;loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape=loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_1*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
}
;loss/dense_5_loss/softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
и
9loss/dense_5_loss/softmax_cross_entropy_with_logits/Sub_2Sub8loss/dense_5_loss/softmax_cross_entropy_with_logits/Rank;loss/dense_5_loss/softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 

Aloss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Н
@loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_2/sizePack9loss/dense_5_loss/softmax_cross_entropy_with_logits/Sub_2*
_output_shapes
:*

axis *
T0*
N
Ж
;loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_2Slice9loss/dense_5_loss/softmax_cross_entropy_with_logits/ShapeAloss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_2/begin@loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_2/size*
Index0*
T0*
_output_shapes
:
і
=loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_2Reshape3loss/dense_5_loss/softmax_cross_entropy_with_logits;loss/dense_5_loss/softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
k
&loss/dense_5_loss/weighted_loss/Cast/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

Tloss/dense_5_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
dtype0*
valueB *
_output_shapes
: 

Sloss/dense_5_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
dtype0*
value	B : *
_output_shapes
: 
а
Sloss/dense_5_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape=loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_2*
out_type0*
_output_shapes
:*
T0

Rloss/dense_5_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :
j
bloss/dense_5_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
Ѓ
Aloss/dense_5_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape=loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_2c^loss/dense_5_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
ы
Aloss/dense_5_loss/weighted_loss/broadcast_weights/ones_like/ConstConstc^loss/dense_5_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
valueB
 *  ?*
dtype0

;loss/dense_5_loss/weighted_loss/broadcast_weights/ones_likeFillAloss/dense_5_loss/weighted_loss/broadcast_weights/ones_like/ShapeAloss/dense_5_loss/weighted_loss/broadcast_weights/ones_like/Const*#
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
Ы
1loss/dense_5_loss/weighted_loss/broadcast_weightsMul&loss/dense_5_loss/weighted_loss/Cast/x;loss/dense_5_loss/weighted_loss/broadcast_weights/ones_like*#
_output_shapes
:џџџџџџџџџ*
T0
Ъ
#loss/dense_5_loss/weighted_loss/MulMul=loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_21loss/dense_5_loss/weighted_loss/broadcast_weights*#
_output_shapes
:џџџџџџџџџ*
T0
c
loss/dense_5_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

loss/dense_5_loss/SumSum#loss/dense_5_loss/weighted_loss/Mulloss/dense_5_loss/Const_1*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
|
loss/dense_5_loss/num_elementsSize#loss/dense_5_loss/weighted_loss/Mul*
out_type0*
_output_shapes
: *
T0

#loss/dense_5_loss/num_elements/CastCastloss/dense_5_loss/num_elements*
Truncate( *

SrcT0*

DstT0*
_output_shapes
: 
\
loss/dense_5_loss/Const_2Const*
dtype0*
_output_shapes
: *
valueB 

loss/dense_5_loss/Sum_1Sumloss/dense_5_loss/Sumloss/dense_5_loss/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

loss/dense_5_loss/valueDivNoNanloss/dense_5_loss/Sum_1#loss/dense_5_loss/num_elements/Cast*
_output_shapes
: *
T0
O

loss/mul/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
U
loss/mulMul
loss/mul/xloss/dense_5_loss/value*
T0*
_output_shapes
: 
j
'training/Adam/gradients/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
p
+training/Adam/gradients/gradients/grad_ys_0Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 
З
&training/Adam/gradients/gradients/FillFill'training/Adam/gradients/gradients/Shape+training/Adam/gradients/gradients/grad_ys_0*

index_type0*
T0*
_output_shapes
: 

3training/Adam/gradients/gradients/loss/mul_grad/MulMul&training/Adam/gradients/gradients/Fillloss/dense_5_loss/value*
_output_shapes
: *
T0

5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Mul&training/Adam/gradients/gradients/Fill
loss/mul/x*
T0*
_output_shapes
: 

Dtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

Ftraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
И
Ttraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsDtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/ShapeFtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
в
Itraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/div_no_nanDivNoNan5training/Adam/gradients/gradients/loss/mul_grad/Mul_1#loss/dense_5_loss/num_elements/Cast*
T0*
_output_shapes
: 
Ј
Btraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/SumSumItraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/div_no_nanTtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0

Ftraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/ReshapeReshapeBtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/SumDtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Btraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/NegNegloss/dense_5_loss/Sum_1*
T0*
_output_shapes
: 
с
Ktraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/div_no_nan_1DivNoNanBtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/Neg#loss/dense_5_loss/num_elements/Cast*
T0*
_output_shapes
: 
ъ
Ktraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/div_no_nan_2DivNoNanKtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/div_no_nan_1#loss/dense_5_loss/num_elements/Cast*
T0*
_output_shapes
: 
ю
Btraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/mulMul5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Ktraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/div_no_nan_2*
_output_shapes
: *
T0
Ѕ
Dtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/Sum_1SumBtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/mulVtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

Htraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/Reshape_1ReshapeDtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/Sum_1Ftraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 

Ltraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB 

Ftraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_1_grad/ReshapeReshapeFtraining/Adam/gradients/gradients/loss/dense_5_loss/value_grad/ReshapeLtraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_1_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0

Dtraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_1_grad/ConstConst*
valueB *
_output_shapes
: *
dtype0

Ctraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_1_grad/TileTileFtraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_1_grad/ReshapeDtraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_1_grad/Const*

Tmultiples0*
_output_shapes
: *
T0

Jtraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0

Dtraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_grad/ReshapeReshapeCtraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_1_grad/TileJtraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
Ѕ
Btraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_grad/ShapeShape#loss/dense_5_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
:

Atraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_grad/TileTileDtraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_grad/ReshapeBtraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ
Э
Ptraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/ShapeShape=loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
_output_shapes
:*
out_type0
У
Rtraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/Shape_1Shape1loss/dense_5_loss/weighted_loss/broadcast_weights*
out_type0*
T0*
_output_shapes
:
м
`training/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/ShapeRtraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
Ntraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/MulMulAtraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_grad/Tile1loss/dense_5_loss/weighted_loss/broadcast_weights*#
_output_shapes
:џџџџџџџџџ*
T0
Ч
Ntraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/SumSumNtraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/Mul`training/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
Л
Rtraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/ReshapeReshapeNtraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/SumPtraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

Ptraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/Mul_1Mul=loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_2Atraining/Adam/gradients/gradients/loss/dense_5_loss/Sum_grad/Tile*#
_output_shapes
:џџџџџџџџџ*
T0
Э
Ptraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/Sum_1SumPtraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/Mul_1btraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
С
Ttraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/Reshape_1ReshapePtraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/Sum_1Rtraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:џџџџџџџџџ
н
jtraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape3loss/dense_5_loss/softmax_cross_entropy_with_logits*
T0*
out_type0*
_output_shapes
:
ѓ
ltraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapeRtraining/Adam/gradients/gradients/loss/dense_5_loss/weighted_loss/Mul_grad/Reshapejtraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ћ
,training/Adam/gradients/gradients/zeros_like	ZerosLike5loss/dense_5_loss/softmax_cross_entropy_with_logits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
Д
itraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ

etraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsltraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapeitraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*

Tdim0*'
_output_shapes
:џџџџџџџџџ*
T0
О
^training/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/mulMuletraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/ExpandDims5loss/dense_5_loss/softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ы
etraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax;loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

^training/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/NegNegetraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
Ж
ktraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0

gtraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsltraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapektraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*
T0*'
_output_shapes
:џџџџџџџџџ*

Tdim0
ы
`training/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/mul_1Mulgtraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1^training/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/Neg*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
З
htraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapedense_5/BiasAdd*
_output_shapes
:*
out_type0*
T0
џ
jtraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshape^training/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits_grad/mulhtraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
љ
Btraining/Adam/gradients/gradients/dense_5/BiasAdd_grad/BiasAddGradBiasAddGradjtraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
data_formatNHWC*
T0*
_output_shapes
:
Њ
<training/Adam/gradients/gradients/dense_5/MatMul_grad/MatMulMatMuljtraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshapedense_5/MatMul/ReadVariableOp*(
_output_shapes
:џџџџџџџџџ *
T0*
transpose_a( *
transpose_b(

>training/Adam/gradients/gradients/dense_5/MatMul_grad/MatMul_1MatMuldense_4/Relujtraining/Adam/gradients/gradients/loss/dense_5_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
transpose_b( *
transpose_a(*
_output_shapes
:	 *
T0
Ч
<training/Adam/gradients/gradients/dense_4/Relu_grad/ReluGradReluGrad<training/Adam/gradients/gradients/dense_5/MatMul_grad/MatMuldense_4/Relu*(
_output_shapes
:џџџџџџџџџ *
T0
Ь
Btraining/Adam/gradients/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/gradients/dense_4/Relu_grad/ReluGrad*
_output_shapes	
: *
T0*
data_formatNHWC
ќ
<training/Adam/gradients/gradients/dense_4/MatMul_grad/MatMulMatMul<training/Adam/gradients/gradients/dense_4/Relu_grad/ReluGraddense_4/MatMul/ReadVariableOp*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:џџџџџџџџџ 
ы
>training/Adam/gradients/gradients/dense_4/MatMul_grad/MatMul_1MatMuldropout_2/Identity<training/Adam/gradients/gradients/dense_4/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( * 
_output_shapes
:
  *
T0
Ч
<training/Adam/gradients/gradients/dense_3/Relu_grad/ReluGradReluGrad<training/Adam/gradients/gradients/dense_4/MatMul_grad/MatMuldense_3/Relu*
T0*(
_output_shapes
:џџџџџџџџџ 
Ь
Btraining/Adam/gradients/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/gradients/dense_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
: 
ќ
<training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMulMatMul<training/Adam/gradients/gradients/dense_3/Relu_grad/ReluGraddense_3/MatMul/ReadVariableOp*
transpose_a( *(
_output_shapes
:џџџџџџџџџ *
T0*
transpose_b(
х
>training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu<training/Adam/gradients/gradients/dense_3/Relu_grad/ReluGrad*
transpose_b( * 
_output_shapes
:
  *
T0*
transpose_a(
Ч
<training/Adam/gradients/gradients/dense_2/Relu_grad/ReluGradReluGrad<training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*(
_output_shapes
:џџџџџџџџџ *
T0
Ь
Btraining/Adam/gradients/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/gradients/dense_2/Relu_grad/ReluGrad*
T0*
_output_shapes	
: *
data_formatNHWC
ќ
<training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMulMatMul<training/Adam/gradients/gradients/dense_2/Relu_grad/ReluGraddense_2/MatMul/ReadVariableOp*
transpose_b(*
transpose_a( *(
_output_shapes
:џџџџџџџџџ *
T0
ы
>training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMul_1MatMuldropout_1/Identity<training/Adam/gradients/gradients/dense_2/Relu_grad/ReluGrad*
T0*
transpose_a(*
transpose_b( * 
_output_shapes
:
  
Ч
<training/Adam/gradients/gradients/dense_1/Relu_grad/ReluGradReluGrad<training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMuldense_1/Relu*
T0*(
_output_shapes
:џџџџџџџџџ 
Ь
Btraining/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/gradients/dense_1/Relu_grad/ReluGrad*
_output_shapes	
: *
T0*
data_formatNHWC
§
<training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMulMatMul<training/Adam/gradients/gradients/dense_1/Relu_grad/ReluGraddense_1/MatMul/ReadVariableOp*)
_output_shapes
:џџџџџџџџџ	*
transpose_b(*
transpose_a( *
T0
ы
>training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1MatMulflatten_1/Reshape<training/Adam/gradients/gradients/dense_1/Relu_grad/ReluGrad*
transpose_b( *
T0*
transpose_a(*!
_output_shapes
:	 

$training/Adam/iter/Initializer/zerosConst*
_output_shapes
: *%
_class
loc:@training/Adam/iter*
dtype0	*
value	B	 R 
А
training/Adam/iterVarHandleOp*#
shared_nametraining/Adam/iter*
	container *
dtype0	*%
_class
loc:@training/Adam/iter*
_output_shapes
: *
shape: 
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
dtype0*'
_class
loc:@training/Adam/beta_1*
valueB
 *fff?*
_output_shapes
: 
Ж
training/Adam/beta_1VarHandleOp*'
_class
loc:@training/Adam/beta_1*
_output_shapes
: *
dtype0*%
shared_nametraining/Adam/beta_1*
	container *
shape: 
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
.training/Adam/beta_2/Initializer/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *wО?*'
_class
loc:@training/Adam/beta_2
Ж
training/Adam/beta_2VarHandleOp*'
_class
loc:@training/Adam/beta_2*%
shared_nametraining/Adam/beta_2*
	container *
dtype0*
shape: *
_output_shapes
: 
y
5training/Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 

training/Adam/beta_2/AssignAssignVariableOptraining/Adam/beta_2.training/Adam/beta_2/Initializer/initial_value*
dtype0
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
_output_shapes
: *
dtype0

-training/Adam/decay/Initializer/initial_valueConst*
dtype0*&
_class
loc:@training/Adam/decay*
valueB
 *    *
_output_shapes
: 
Г
training/Adam/decayVarHandleOp*
	container *
_output_shapes
: *$
shared_nametraining/Adam/decay*
shape: *&
_class
loc:@training/Adam/decay*
dtype0
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
5training/Adam/learning_rate/Initializer/initial_valueConst*.
_class$
" loc:@training/Adam/learning_rate*
valueB
 *o:*
dtype0*
_output_shapes
: 
Ы
training/Adam/learning_rateVarHandleOp*.
_class$
" loc:@training/Adam/learning_rate*,
shared_nametraining/Adam/learning_rate*
dtype0*
	container *
shape: *
_output_shapes
: 
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
Д
@training/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_1/kernel*
_output_shapes
:*
dtype0*
valueB" L    

6training/Adam/dense_1/kernel/m/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

0training/Adam/dense_1/kernel/m/Initializer/zerosFill@training/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_1/kernel/m/Initializer/zeros/Const*

index_type0*!
_output_shapes
:	 *
T0*!
_class
loc:@dense_1/kernel
Я
training/Adam/dense_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: */
shared_name training/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
shape:	 *
	container 
А
?training/Adam/dense_1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/kernel/m*
_output_shapes
: *!
_class
loc:@dense_1/kernel

%training/Adam/dense_1/kernel/m/AssignAssignVariableOptraining/Adam/dense_1/kernel/m0training/Adam/dense_1/kernel/m/Initializer/zeros*
dtype0
З
2training/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/m*!
_output_shapes
:	 *!
_class
loc:@dense_1/kernel*
dtype0
Њ
>training/Adam/dense_1/bias/m/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@dense_1/bias*
valueB: *
dtype0

4training/Adam/dense_1/bias/m/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@dense_1/bias*
dtype0

.training/Adam/dense_1/bias/m/Initializer/zerosFill>training/Adam/dense_1/bias/m/Initializer/zeros/shape_as_tensor4training/Adam/dense_1/bias/m/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense_1/bias*
_output_shapes	
: 
У
training/Adam/dense_1/bias/mVarHandleOp*-
shared_nametraining/Adam/dense_1/bias/m*
	container *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: *
shape: 
Њ
=training/Adam/dense_1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/bias/m*
_class
loc:@dense_1/bias*
_output_shapes
: 

#training/Adam/dense_1/bias/m/AssignAssignVariableOptraining/Adam/dense_1/bias/m.training/Adam/dense_1/bias/m/Initializer/zeros*
dtype0
Ћ
0training/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/m*
_output_shapes	
: *
_class
loc:@dense_1/bias*
dtype0
Д
@training/Adam/dense_2/kernel/m/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      *!
_class
loc:@dense_2/kernel

6training/Adam/dense_2/kernel/m/Initializer/zeros/ConstConst*
dtype0*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
valueB
 *    

0training/Adam/dense_2/kernel/m/Initializer/zerosFill@training/Adam/dense_2/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_2/kernel/m/Initializer/zeros/Const*

index_type0*!
_class
loc:@dense_2/kernel*
T0* 
_output_shapes
:
  
Ю
training/Adam/dense_2/kernel/mVarHandleOp*
dtype0*
shape:
  *
	container *!
_class
loc:@dense_2/kernel*/
shared_name training/Adam/dense_2/kernel/m*
_output_shapes
: 
А
?training/Adam/dense_2/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_2/kernel/m*
_output_shapes
: *!
_class
loc:@dense_2/kernel
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
dtype0*
valueB: *
_class
loc:@dense_2/bias*
_output_shapes
:

4training/Adam/dense_2/bias/m/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0*
_class
loc:@dense_2/bias

.training/Adam/dense_2/bias/m/Initializer/zerosFill>training/Adam/dense_2/bias/m/Initializer/zeros/shape_as_tensor4training/Adam/dense_2/bias/m/Initializer/zeros/Const*

index_type0*
T0*
_output_shapes	
: *
_class
loc:@dense_2/bias
У
training/Adam/dense_2/bias/mVarHandleOp*
_class
loc:@dense_2/bias*
_output_shapes
: *
dtype0*-
shared_nametraining/Adam/dense_2/bias/m*
shape: *
	container 
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
dtype0*
_output_shapes	
: *
_class
loc:@dense_2/bias
Д
@training/Adam/dense_3/kernel/m/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
:

6training/Adam/dense_3/kernel/m/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0*!
_class
loc:@dense_3/kernel

0training/Adam/dense_3/kernel/m/Initializer/zerosFill@training/Adam/dense_3/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_3/kernel/m/Initializer/zeros/Const*

index_type0*!
_class
loc:@dense_3/kernel*
T0* 
_output_shapes
:
  
Ю
training/Adam/dense_3/kernel/mVarHandleOp*!
_class
loc:@dense_3/kernel*/
shared_name training/Adam/dense_3/kernel/m*
_output_shapes
: *
dtype0*
	container *
shape:
  
А
?training/Adam/dense_3/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/kernel/m*
_output_shapes
: *!
_class
loc:@dense_3/kernel

%training/Adam/dense_3/kernel/m/AssignAssignVariableOptraining/Adam/dense_3/kernel/m0training/Adam/dense_3/kernel/m/Initializer/zeros*
dtype0
Ж
2training/Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/kernel/m*
dtype0*!
_class
loc:@dense_3/kernel* 
_output_shapes
:
  
Њ
>training/Adam/dense_3/bias/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB: *
_class
loc:@dense_3/bias*
_output_shapes
:

4training/Adam/dense_3/bias/m/Initializer/zeros/ConstConst*
_class
loc:@dense_3/bias*
valueB
 *    *
_output_shapes
: *
dtype0

.training/Adam/dense_3/bias/m/Initializer/zerosFill>training/Adam/dense_3/bias/m/Initializer/zeros/shape_as_tensor4training/Adam/dense_3/bias/m/Initializer/zeros/Const*
_output_shapes	
: *
_class
loc:@dense_3/bias*

index_type0*
T0
У
training/Adam/dense_3/bias/mVarHandleOp*-
shared_nametraining/Adam/dense_3/bias/m*
	container *
_output_shapes
: *
dtype0*
_class
loc:@dense_3/bias*
shape: 
Њ
=training/Adam/dense_3/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/bias/m*
_output_shapes
: *
_class
loc:@dense_3/bias

#training/Adam/dense_3/bias/m/AssignAssignVariableOptraining/Adam/dense_3/bias/m.training/Adam/dense_3/bias/m/Initializer/zeros*
dtype0
Ћ
0training/Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/bias/m*
dtype0*
_output_shapes	
: *
_class
loc:@dense_3/bias
Д
@training/Adam/dense_4/kernel/m/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_4/kernel*
valueB"      *
_output_shapes
:*
dtype0

6training/Adam/dense_4/kernel/m/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*!
_class
loc:@dense_4/kernel*
valueB
 *    

0training/Adam/dense_4/kernel/m/Initializer/zerosFill@training/Adam/dense_4/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_4/kernel/m/Initializer/zeros/Const*

index_type0*!
_class
loc:@dense_4/kernel*
T0* 
_output_shapes
:
  
Ю
training/Adam/dense_4/kernel/mVarHandleOp*
dtype0*/
shared_name training/Adam/dense_4/kernel/m*
	container *
_output_shapes
: *!
_class
loc:@dense_4/kernel*
shape:
  
А
?training/Adam/dense_4/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_4/kernel/m*!
_class
loc:@dense_4/kernel*
_output_shapes
: 

%training/Adam/dense_4/kernel/m/AssignAssignVariableOptraining/Adam/dense_4/kernel/m0training/Adam/dense_4/kernel/m/Initializer/zeros*
dtype0
Ж
2training/Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_4/kernel/m*!
_class
loc:@dense_4/kernel*
dtype0* 
_output_shapes
:
  
Њ
>training/Adam/dense_4/bias/m/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense_4/bias*
dtype0*
valueB: *
_output_shapes
:

4training/Adam/dense_4/bias/m/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *
_class
loc:@dense_4/bias*
_output_shapes
: 

.training/Adam/dense_4/bias/m/Initializer/zerosFill>training/Adam/dense_4/bias/m/Initializer/zeros/shape_as_tensor4training/Adam/dense_4/bias/m/Initializer/zeros/Const*
_class
loc:@dense_4/bias*

index_type0*
_output_shapes	
: *
T0
У
training/Adam/dense_4/bias/mVarHandleOp*
shape: *
_output_shapes
: *
_class
loc:@dense_4/bias*
	container *
dtype0*-
shared_nametraining/Adam/dense_4/bias/m
Њ
=training/Adam/dense_4/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_4/bias/m*
_class
loc:@dense_4/bias*
_output_shapes
: 

#training/Adam/dense_4/bias/m/AssignAssignVariableOptraining/Adam/dense_4/bias/m.training/Adam/dense_4/bias/m/Initializer/zeros*
dtype0
Ћ
0training/Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_4/bias/m*
_output_shapes	
: *
_class
loc:@dense_4/bias*
dtype0
Д
@training/Adam/dense_5/kernel/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"      *
_output_shapes
:*!
_class
loc:@dense_5/kernel

6training/Adam/dense_5/kernel/m/Initializer/zeros/ConstConst*
dtype0*!
_class
loc:@dense_5/kernel*
_output_shapes
: *
valueB
 *    

0training/Adam/dense_5/kernel/m/Initializer/zerosFill@training/Adam/dense_5/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_5/kernel/m/Initializer/zeros/Const*

index_type0*
_output_shapes
:	 *!
_class
loc:@dense_5/kernel*
T0
Э
training/Adam/dense_5/kernel/mVarHandleOp*
shape:	 *
dtype0*!
_class
loc:@dense_5/kernel*
_output_shapes
: *
	container */
shared_name training/Adam/dense_5/kernel/m
А
?training/Adam/dense_5/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_5/kernel/m*
_output_shapes
: *!
_class
loc:@dense_5/kernel

%training/Adam/dense_5/kernel/m/AssignAssignVariableOptraining/Adam/dense_5/kernel/m0training/Adam/dense_5/kernel/m/Initializer/zeros*
dtype0
Е
2training/Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_5/kernel/m*
dtype0*
_output_shapes
:	 *!
_class
loc:@dense_5/kernel

.training/Adam/dense_5/bias/m/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@dense_5/bias*
valueB*    *
dtype0
Т
training/Adam/dense_5/bias/mVarHandleOp*
shape:*
_class
loc:@dense_5/bias*
_output_shapes
: *-
shared_nametraining/Adam/dense_5/bias/m*
	container *
dtype0
Њ
=training/Adam/dense_5/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_5/bias/m*
_class
loc:@dense_5/bias*
_output_shapes
: 

#training/Adam/dense_5/bias/m/AssignAssignVariableOptraining/Adam/dense_5/bias/m.training/Adam/dense_5/bias/m/Initializer/zeros*
dtype0
Њ
0training/Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_5/bias/m*
dtype0*
_class
loc:@dense_5/bias*
_output_shapes
:
Д
@training/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*!
_class
loc:@dense_1/kernel*
valueB" L    

6training/Adam/dense_1/kernel/v/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: *!
_class
loc:@dense_1/kernel

0training/Adam/dense_1/kernel/v/Initializer/zerosFill@training/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_1/kernel/v/Initializer/zeros/Const*
T0*!
_output_shapes
:	 *!
_class
loc:@dense_1/kernel*

index_type0
Я
training/Adam/dense_1/kernel/vVarHandleOp*
	container */
shared_name training/Adam/dense_1/kernel/v*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
shape:	 *
dtype0
А
?training/Adam/dense_1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/kernel/v*!
_class
loc:@dense_1/kernel*
_output_shapes
: 

%training/Adam/dense_1/kernel/v/AssignAssignVariableOptraining/Adam/dense_1/kernel/v0training/Adam/dense_1/kernel/v/Initializer/zeros*
dtype0
З
2training/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/v*!
_output_shapes
:	 *!
_class
loc:@dense_1/kernel*
dtype0
Њ
>training/Adam/dense_1/bias/v/Initializer/zeros/shape_as_tensorConst*
valueB: *
_class
loc:@dense_1/bias*
_output_shapes
:*
dtype0

4training/Adam/dense_1/bias/v/Initializer/zeros/ConstConst*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: *
valueB
 *    

.training/Adam/dense_1/bias/v/Initializer/zerosFill>training/Adam/dense_1/bias/v/Initializer/zeros/shape_as_tensor4training/Adam/dense_1/bias/v/Initializer/zeros/Const*
T0*
_output_shapes	
: *
_class
loc:@dense_1/bias*

index_type0
У
training/Adam/dense_1/bias/vVarHandleOp*
dtype0*
	container *
_class
loc:@dense_1/bias*
shape: *-
shared_nametraining/Adam/dense_1/bias/v*
_output_shapes
: 
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
loc:@dense_1/bias*
_output_shapes	
: *
dtype0
Д
@training/Adam/dense_2/kernel/v/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"      *!
_class
loc:@dense_2/kernel*
dtype0

6training/Adam/dense_2/kernel/v/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: *!
_class
loc:@dense_2/kernel

0training/Adam/dense_2/kernel/v/Initializer/zerosFill@training/Adam/dense_2/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_2/kernel/v/Initializer/zeros/Const*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
  *

index_type0*
T0
Ю
training/Adam/dense_2/kernel/vVarHandleOp*
	container *
dtype0*!
_class
loc:@dense_2/kernel*
shape:
  */
shared_name training/Adam/dense_2/kernel/v*
_output_shapes
: 
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
_output_shapes
:
  *
dtype0
Њ
>training/Adam/dense_2/bias/v/Initializer/zeros/shape_as_tensorConst*
valueB: *
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
:

4training/Adam/dense_2/bias/v/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@dense_2/bias

.training/Adam/dense_2/bias/v/Initializer/zerosFill>training/Adam/dense_2/bias/v/Initializer/zeros/shape_as_tensor4training/Adam/dense_2/bias/v/Initializer/zeros/Const*
_class
loc:@dense_2/bias*
T0*

index_type0*
_output_shapes	
: 
У
training/Adam/dense_2/bias/vVarHandleOp*
_class
loc:@dense_2/bias*-
shared_nametraining/Adam/dense_2/bias/v*
	container *
dtype0*
shape: *
_output_shapes
: 
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
@training/Adam/dense_3/kernel/v/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
:*
valueB"      

6training/Adam/dense_3/kernel/v/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*!
_class
loc:@dense_3/kernel*
_output_shapes
: 

0training/Adam/dense_3/kernel/v/Initializer/zerosFill@training/Adam/dense_3/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_3/kernel/v/Initializer/zeros/Const* 
_output_shapes
:
  *!
_class
loc:@dense_3/kernel*

index_type0*
T0
Ю
training/Adam/dense_3/kernel/vVarHandleOp*
	container *
_output_shapes
: */
shared_name training/Adam/dense_3/kernel/v*
shape:
  *!
_class
loc:@dense_3/kernel*
dtype0
А
?training/Adam/dense_3/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/kernel/v*!
_class
loc:@dense_3/kernel*
_output_shapes
: 

%training/Adam/dense_3/kernel/v/AssignAssignVariableOptraining/Adam/dense_3/kernel/v0training/Adam/dense_3/kernel/v/Initializer/zeros*
dtype0
Ж
2training/Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/kernel/v*!
_class
loc:@dense_3/kernel*
dtype0* 
_output_shapes
:
  
Њ
>training/Adam/dense_3/bias/v/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB: *
_class
loc:@dense_3/bias*
_output_shapes
:

4training/Adam/dense_3/bias/v/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 

.training/Adam/dense_3/bias/v/Initializer/zerosFill>training/Adam/dense_3/bias/v/Initializer/zeros/shape_as_tensor4training/Adam/dense_3/bias/v/Initializer/zeros/Const*

index_type0*
_output_shapes	
: *
T0*
_class
loc:@dense_3/bias
У
training/Adam/dense_3/bias/vVarHandleOp*
	container *
_output_shapes
: *-
shared_nametraining/Adam/dense_3/bias/v*
shape: *
dtype0*
_class
loc:@dense_3/bias
Њ
=training/Adam/dense_3/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/bias/v*
_output_shapes
: *
_class
loc:@dense_3/bias

#training/Adam/dense_3/bias/v/AssignAssignVariableOptraining/Adam/dense_3/bias/v.training/Adam/dense_3/bias/v/Initializer/zeros*
dtype0
Ћ
0training/Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/bias/v*
_class
loc:@dense_3/bias*
_output_shapes	
: *
dtype0
Д
@training/Adam/dense_4/kernel/v/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"      *
dtype0*!
_class
loc:@dense_4/kernel

6training/Adam/dense_4/kernel/v/Initializer/zeros/ConstConst*!
_class
loc:@dense_4/kernel*
dtype0*
valueB
 *    *
_output_shapes
: 

0training/Adam/dense_4/kernel/v/Initializer/zerosFill@training/Adam/dense_4/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_4/kernel/v/Initializer/zeros/Const*!
_class
loc:@dense_4/kernel*

index_type0*
T0* 
_output_shapes
:
  
Ю
training/Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
shape:
  */
shared_name training/Adam/dense_4/kernel/v*!
_class
loc:@dense_4/kernel*
dtype0*
	container 
А
?training/Adam/dense_4/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_4/kernel/v*
_output_shapes
: *!
_class
loc:@dense_4/kernel

%training/Adam/dense_4/kernel/v/AssignAssignVariableOptraining/Adam/dense_4/kernel/v0training/Adam/dense_4/kernel/v/Initializer/zeros*
dtype0
Ж
2training/Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_4/kernel/v* 
_output_shapes
:
  *
dtype0*!
_class
loc:@dense_4/kernel
Њ
>training/Adam/dense_4/bias/v/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB: *
_output_shapes
:*
_class
loc:@dense_4/bias

4training/Adam/dense_4/bias/v/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@dense_4/bias*
valueB
 *    

.training/Adam/dense_4/bias/v/Initializer/zerosFill>training/Adam/dense_4/bias/v/Initializer/zeros/shape_as_tensor4training/Adam/dense_4/bias/v/Initializer/zeros/Const*
_output_shapes	
: *
_class
loc:@dense_4/bias*
T0*

index_type0
У
training/Adam/dense_4/bias/vVarHandleOp*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
: *
shape: *
	container *-
shared_nametraining/Adam/dense_4/bias/v
Њ
=training/Adam/dense_4/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_4/bias/v*
_output_shapes
: *
_class
loc:@dense_4/bias

#training/Adam/dense_4/bias/v/AssignAssignVariableOptraining/Adam/dense_4/bias/v.training/Adam/dense_4/bias/v/Initializer/zeros*
dtype0
Ћ
0training/Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_4/bias/v*
_output_shapes	
: *
dtype0*
_class
loc:@dense_4/bias
Д
@training/Adam/dense_5/kernel/v/Initializer/zeros/shape_as_tensorConst*
valueB"      *
dtype0*!
_class
loc:@dense_5/kernel*
_output_shapes
:

6training/Adam/dense_5/kernel/v/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *!
_class
loc:@dense_5/kernel

0training/Adam/dense_5/kernel/v/Initializer/zerosFill@training/Adam/dense_5/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_5/kernel/v/Initializer/zeros/Const*!
_class
loc:@dense_5/kernel*
_output_shapes
:	 *

index_type0*
T0
Э
training/Adam/dense_5/kernel/vVarHandleOp*!
_class
loc:@dense_5/kernel*
_output_shapes
: *
shape:	 */
shared_name training/Adam/dense_5/kernel/v*
dtype0*
	container 
А
?training/Adam/dense_5/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_5/kernel/v*
_output_shapes
: *!
_class
loc:@dense_5/kernel

%training/Adam/dense_5/kernel/v/AssignAssignVariableOptraining/Adam/dense_5/kernel/v0training/Adam/dense_5/kernel/v/Initializer/zeros*
dtype0
Е
2training/Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_5/kernel/v*
_output_shapes
:	 *!
_class
loc:@dense_5/kernel*
dtype0

.training/Adam/dense_5/bias/v/Initializer/zerosConst*
_class
loc:@dense_5/bias*
dtype0*
_output_shapes
:*
valueB*    
Т
training/Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *-
shared_nametraining/Adam/dense_5/bias/v*
dtype0*
_class
loc:@dense_5/bias*
	container *
shape:
Њ
=training/Adam/dense_5/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_5/bias/v*
_output_shapes
: *
_class
loc:@dense_5/bias

#training/Adam/dense_5/bias/v/AssignAssignVariableOptraining/Adam/dense_5/bias/v.training/Adam/dense_5/bias/v/Initializer/zeros*
dtype0
Њ
0training/Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_5/bias/v*
dtype0*
_output_shapes
:*
_class
loc:@dense_5/bias
y
%training/Adam/Identity/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
_output_shapes
: *
dtype0
j
training/Adam/IdentityIdentity%training/Adam/Identity/ReadVariableOp*
_output_shapes
: *
T0
g
training/Adam/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 
U
training/Adam/add/yConst*
dtype0	*
value	B	 R*
_output_shapes
: 
n
training/Adam/addAddV2training/Adam/ReadVariableOptraining/Adam/add/y*
T0	*
_output_shapes
: 
m
training/Adam/CastCasttraining/Adam/add*
Truncate( *

SrcT0	*
_output_shapes
: *

DstT0
t
'training/Adam/Identity_1/ReadVariableOpReadVariableOptraining/Adam/beta_1*
_output_shapes
: *
dtype0
n
training/Adam/Identity_1Identity'training/Adam/Identity_1/ReadVariableOp*
_output_shapes
: *
T0
t
'training/Adam/Identity_2/ReadVariableOpReadVariableOptraining/Adam/beta_2*
_output_shapes
: *
dtype0
n
training/Adam/Identity_2Identity'training/Adam/Identity_2/ReadVariableOp*
_output_shapes
: *
T0
g
training/Adam/PowPowtraining/Adam/Identity_1training/Adam/Cast*
_output_shapes
: *
T0
i
training/Adam/Pow_1Powtraining/Adam/Identity_2training/Adam/Cast*
_output_shapes
: *
T0
X
training/Adam/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
c
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
N
training/Adam/SqrtSqrttraining/Adam/sub*
_output_shapes
: *
T0
Z
training/Adam/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
e
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
h
training/Adam/mulMultraining/Adam/Identitytraining/Adam/truediv*
T0*
_output_shapes
: 
X
training/Adam/ConstConst*
dtype0*
valueB
 *wЬ+2*
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
training/Adam/sub_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
l
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/Identity_2*
_output_shapes
: *
T0
Л
:training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdamResourceApplyAdamdense_1/kerneltraining/Adam/dense_1/kernel/mtraining/Adam/dense_1/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1*
use_locking(*
use_nesterov( *
T0*!
_class
loc:@dense_1/kernel
Е
8training/Adam/Adam/update_dense_1/bias/ResourceApplyAdamResourceApplyAdamdense_1/biastraining/Adam/dense_1/bias/mtraining/Adam/dense_1/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
use_nesterov( *
_class
loc:@dense_1/bias
Л
:training/Adam/Adam/update_dense_2/kernel/ResourceApplyAdamResourceApplyAdamdense_2/kerneltraining/Adam/dense_2/kernel/mtraining/Adam/dense_2/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMul_1*
use_locking(*!
_class
loc:@dense_2/kernel*
T0*
use_nesterov( 
Е
8training/Adam/Adam/update_dense_2/bias/ResourceApplyAdamResourceApplyAdamdense_2/biastraining/Adam/dense_2/bias/mtraining/Adam/dense_2/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_class
loc:@dense_2/bias*
T0*
use_locking(*
use_nesterov( 
Л
:training/Adam/Adam/update_dense_3/kernel/ResourceApplyAdamResourceApplyAdamdense_3/kerneltraining/Adam/dense_3/kernel/mtraining/Adam/dense_3/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMul_1*
use_locking(*
use_nesterov( *
T0*!
_class
loc:@dense_3/kernel
Е
8training/Adam/Adam/update_dense_3/bias/ResourceApplyAdamResourceApplyAdamdense_3/biastraining/Adam/dense_3/bias/mtraining/Adam/dense_3/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
use_locking(*
use_nesterov( *
_class
loc:@dense_3/bias*
T0
Л
:training/Adam/Adam/update_dense_4/kernel/ResourceApplyAdamResourceApplyAdamdense_4/kerneltraining/Adam/dense_4/kernel/mtraining/Adam/dense_4/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_4/MatMul_grad/MatMul_1*!
_class
loc:@dense_4/kernel*
use_locking(*
T0*
use_nesterov( 
Е
8training/Adam/Adam/update_dense_4/bias/ResourceApplyAdamResourceApplyAdamdense_4/biastraining/Adam/dense_4/bias/mtraining/Adam/dense_4/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
use_nesterov( *
T0*
_class
loc:@dense_4/bias*
use_locking(
Л
:training/Adam/Adam/update_dense_5/kernel/ResourceApplyAdamResourceApplyAdamdense_5/kerneltraining/Adam/dense_5/kernel/mtraining/Adam/dense_5/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_5/MatMul_grad/MatMul_1*
use_locking(*
use_nesterov( *!
_class
loc:@dense_5/kernel*
T0
Е
8training/Adam/Adam/update_dense_5/bias/ResourceApplyAdamResourceApplyAdamdense_5/biastraining/Adam/dense_5/bias/mtraining/Adam/dense_5/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
T0*
_class
loc:@dense_5/bias*
use_locking(*
use_nesterov( 
В
training/Adam/Adam/ConstConst9^training/Adam/Adam/update_dense_1/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_2/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_2/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_3/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_3/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_4/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_4/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_5/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_5/kernel/ResourceApplyAdam*
dtype0	*
value	B	 R*
_output_shapes
: 
x
&training/Adam/Adam/AssignAddVariableOpAssignAddVariableOptraining/Adam/itertraining/Adam/Adam/Const*
dtype0	
э
!training/Adam/Adam/ReadVariableOpReadVariableOptraining/Adam/iter'^training/Adam/Adam/AssignAddVariableOp9^training/Adam/Adam/update_dense_1/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_2/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_2/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_3/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_3/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_4/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_4/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_5/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_5/kernel/ResourceApplyAdam*
_output_shapes
: *
dtype0	
Q
training_1/group_depsNoOp	^loss/mul'^training/Adam/Adam/AssignAddVariableOp
c
VarIsInitializedOp_10VarIsInitializedOptraining/Adam/dense_1/kernel/v*
_output_shapes
: 
Y
VarIsInitializedOp_11VarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 
Y
VarIsInitializedOp_12VarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 
`
VarIsInitializedOp_13VarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 
c
VarIsInitializedOp_14VarIsInitializedOptraining/Adam/dense_4/kernel/m*
_output_shapes
: 
a
VarIsInitializedOp_15VarIsInitializedOptraining/Adam/dense_4/bias/m*
_output_shapes
: 
a
VarIsInitializedOp_16VarIsInitializedOptraining/Adam/dense_1/bias/v*
_output_shapes
: 
J
VarIsInitializedOp_17VarIsInitializedOptotal*
_output_shapes
: 
c
VarIsInitializedOp_18VarIsInitializedOptraining/Adam/dense_5/kernel/m*
_output_shapes
: 
c
VarIsInitializedOp_19VarIsInitializedOptraining/Adam/dense_2/kernel/v*
_output_shapes
: 
a
VarIsInitializedOp_20VarIsInitializedOptraining/Adam/dense_2/bias/v*
_output_shapes
: 
c
VarIsInitializedOp_21VarIsInitializedOptraining/Adam/dense_4/kernel/v*
_output_shapes
: 
a
VarIsInitializedOp_22VarIsInitializedOptraining/Adam/dense_4/bias/v*
_output_shapes
: 
J
VarIsInitializedOp_23VarIsInitializedOpcount*
_output_shapes
: 
X
VarIsInitializedOp_24VarIsInitializedOptraining/Adam/decay*
_output_shapes
: 
c
VarIsInitializedOp_25VarIsInitializedOptraining/Adam/dense_1/kernel/m*
_output_shapes
: 
W
VarIsInitializedOp_26VarIsInitializedOptraining/Adam/iter*
_output_shapes
: 
c
VarIsInitializedOp_27VarIsInitializedOptraining/Adam/dense_2/kernel/m*
_output_shapes
: 
a
VarIsInitializedOp_28VarIsInitializedOptraining/Adam/dense_2/bias/m*
_output_shapes
: 
a
VarIsInitializedOp_29VarIsInitializedOptraining/Adam/dense_3/bias/v*
_output_shapes
: 
c
VarIsInitializedOp_30VarIsInitializedOptraining/Adam/dense_5/kernel/v*
_output_shapes
: 
a
VarIsInitializedOp_31VarIsInitializedOptraining/Adam/dense_5/bias/v*
_output_shapes
: 
a
VarIsInitializedOp_32VarIsInitializedOptraining/Adam/dense_1/bias/m*
_output_shapes
: 
c
VarIsInitializedOp_33VarIsInitializedOptraining/Adam/dense_3/kernel/m*
_output_shapes
: 
a
VarIsInitializedOp_34VarIsInitializedOptraining/Adam/dense_3/bias/m*
_output_shapes
: 
c
VarIsInitializedOp_35VarIsInitializedOptraining/Adam/dense_3/kernel/v*
_output_shapes
: 
a
VarIsInitializedOp_36VarIsInitializedOptraining/Adam/dense_5/bias/m*
_output_shapes
: 
в
init_1NoOp^count/Assign^total/Assign^training/Adam/beta_1/Assign^training/Adam/beta_2/Assign^training/Adam/decay/Assign$^training/Adam/dense_1/bias/m/Assign$^training/Adam/dense_1/bias/v/Assign&^training/Adam/dense_1/kernel/m/Assign&^training/Adam/dense_1/kernel/v/Assign$^training/Adam/dense_2/bias/m/Assign$^training/Adam/dense_2/bias/v/Assign&^training/Adam/dense_2/kernel/m/Assign&^training/Adam/dense_2/kernel/v/Assign$^training/Adam/dense_3/bias/m/Assign$^training/Adam/dense_3/bias/v/Assign&^training/Adam/dense_3/kernel/m/Assign&^training/Adam/dense_3/kernel/v/Assign$^training/Adam/dense_4/bias/m/Assign$^training/Adam/dense_4/bias/v/Assign&^training/Adam/dense_4/kernel/m/Assign&^training/Adam/dense_4/kernel/v/Assign$^training/Adam/dense_5/bias/m/Assign$^training/Adam/dense_5/bias/v/Assign&^training/Adam/dense_5/kernel/m/Assign&^training/Adam/dense_5/kernel/v/Assign^training/Adam/iter/Assign#^training/Adam/learning_rate/Assign
O
Placeholder_10Placeholder*
dtype0	*
shape: *
_output_shapes
: 
X
AssignVariableOp_10AssignVariableOptraining/Adam/iterPlaceholder_10*
dtype0	
r
ReadVariableOp_10ReadVariableOptraining/Adam/iter^AssignVariableOp_10*
dtype0	*
_output_shapes
: 

Placeholder_11Placeholder*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ*
dtype0
d
AssignVariableOp_11AssignVariableOptraining/Adam/dense_1/kernel/mPlaceholder_11*
dtype0

ReadVariableOp_11ReadVariableOptraining/Adam/dense_1/kernel/m^AssignVariableOp_11*!
_output_shapes
:	 *
dtype0
i
Placeholder_12Placeholder*
shape:џџџџџџџџџ*#
_output_shapes
:џџџџџџџџџ*
dtype0
b
AssignVariableOp_12AssignVariableOptraining/Adam/dense_1/bias/mPlaceholder_12*
dtype0

ReadVariableOp_12ReadVariableOptraining/Adam/dense_1/bias/m^AssignVariableOp_12*
dtype0*
_output_shapes	
: 

Placeholder_13Placeholder*%
shape:џџџџџџџџџџџџџџџџџџ*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
d
AssignVariableOp_13AssignVariableOptraining/Adam/dense_2/kernel/mPlaceholder_13*
dtype0

ReadVariableOp_13ReadVariableOptraining/Adam/dense_2/kernel/m^AssignVariableOp_13*
dtype0* 
_output_shapes
:
  
i
Placeholder_14Placeholder*
shape:џџџџџџџџџ*#
_output_shapes
:џџџџџџџџџ*
dtype0
b
AssignVariableOp_14AssignVariableOptraining/Adam/dense_2/bias/mPlaceholder_14*
dtype0

ReadVariableOp_14ReadVariableOptraining/Adam/dense_2/bias/m^AssignVariableOp_14*
dtype0*
_output_shapes	
: 

Placeholder_15Placeholder*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0
d
AssignVariableOp_15AssignVariableOptraining/Adam/dense_3/kernel/mPlaceholder_15*
dtype0

ReadVariableOp_15ReadVariableOptraining/Adam/dense_3/kernel/m^AssignVariableOp_15*
dtype0* 
_output_shapes
:
  
i
Placeholder_16Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
b
AssignVariableOp_16AssignVariableOptraining/Adam/dense_3/bias/mPlaceholder_16*
dtype0

ReadVariableOp_16ReadVariableOptraining/Adam/dense_3/bias/m^AssignVariableOp_16*
dtype0*
_output_shapes	
: 

Placeholder_17Placeholder*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0
d
AssignVariableOp_17AssignVariableOptraining/Adam/dense_4/kernel/mPlaceholder_17*
dtype0

ReadVariableOp_17ReadVariableOptraining/Adam/dense_4/kernel/m^AssignVariableOp_17*
dtype0* 
_output_shapes
:
  
i
Placeholder_18Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
b
AssignVariableOp_18AssignVariableOptraining/Adam/dense_4/bias/mPlaceholder_18*
dtype0

ReadVariableOp_18ReadVariableOptraining/Adam/dense_4/bias/m^AssignVariableOp_18*
dtype0*
_output_shapes	
: 

Placeholder_19Placeholder*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0
d
AssignVariableOp_19AssignVariableOptraining/Adam/dense_5/kernel/mPlaceholder_19*
dtype0

ReadVariableOp_19ReadVariableOptraining/Adam/dense_5/kernel/m^AssignVariableOp_19*
dtype0*
_output_shapes
:	 
i
Placeholder_20Placeholder*
shape:џџџџџџџџџ*#
_output_shapes
:џџџџџџџџџ*
dtype0
b
AssignVariableOp_20AssignVariableOptraining/Adam/dense_5/bias/mPlaceholder_20*
dtype0

ReadVariableOp_20ReadVariableOptraining/Adam/dense_5/bias/m^AssignVariableOp_20*
dtype0*
_output_shapes
:

Placeholder_21Placeholder*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0
d
AssignVariableOp_21AssignVariableOptraining/Adam/dense_1/kernel/vPlaceholder_21*
dtype0

ReadVariableOp_21ReadVariableOptraining/Adam/dense_1/kernel/v^AssignVariableOp_21*!
_output_shapes
:	 *
dtype0
i
Placeholder_22Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
b
AssignVariableOp_22AssignVariableOptraining/Adam/dense_1/bias/vPlaceholder_22*
dtype0

ReadVariableOp_22ReadVariableOptraining/Adam/dense_1/bias/v^AssignVariableOp_22*
_output_shapes	
: *
dtype0

Placeholder_23Placeholder*
dtype0*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
d
AssignVariableOp_23AssignVariableOptraining/Adam/dense_2/kernel/vPlaceholder_23*
dtype0

ReadVariableOp_23ReadVariableOptraining/Adam/dense_2/kernel/v^AssignVariableOp_23*
dtype0* 
_output_shapes
:
  
i
Placeholder_24Placeholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
b
AssignVariableOp_24AssignVariableOptraining/Adam/dense_2/bias/vPlaceholder_24*
dtype0

ReadVariableOp_24ReadVariableOptraining/Adam/dense_2/bias/v^AssignVariableOp_24*
dtype0*
_output_shapes	
: 

Placeholder_25Placeholder*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*%
shape:џџџџџџџџџџџџџџџџџџ
d
AssignVariableOp_25AssignVariableOptraining/Adam/dense_3/kernel/vPlaceholder_25*
dtype0

ReadVariableOp_25ReadVariableOptraining/Adam/dense_3/kernel/v^AssignVariableOp_25*
dtype0* 
_output_shapes
:
  
i
Placeholder_26Placeholder*
dtype0*
shape:џџџџџџџџџ*#
_output_shapes
:џџџџџџџџџ
b
AssignVariableOp_26AssignVariableOptraining/Adam/dense_3/bias/vPlaceholder_26*
dtype0

ReadVariableOp_26ReadVariableOptraining/Adam/dense_3/bias/v^AssignVariableOp_26*
dtype0*
_output_shapes	
: 

Placeholder_27Placeholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
d
AssignVariableOp_27AssignVariableOptraining/Adam/dense_4/kernel/vPlaceholder_27*
dtype0

ReadVariableOp_27ReadVariableOptraining/Adam/dense_4/kernel/v^AssignVariableOp_27*
dtype0* 
_output_shapes
:
  
i
Placeholder_28Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
b
AssignVariableOp_28AssignVariableOptraining/Adam/dense_4/bias/vPlaceholder_28*
dtype0

ReadVariableOp_28ReadVariableOptraining/Adam/dense_4/bias/v^AssignVariableOp_28*
dtype0*
_output_shapes	
: 

Placeholder_29Placeholder*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0
d
AssignVariableOp_29AssignVariableOptraining/Adam/dense_5/kernel/vPlaceholder_29*
dtype0

ReadVariableOp_29ReadVariableOptraining/Adam/dense_5/kernel/v^AssignVariableOp_29*
_output_shapes
:	 *
dtype0
i
Placeholder_30Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
b
AssignVariableOp_30AssignVariableOptraining/Adam/dense_5/bias/vPlaceholder_30*
dtype0

ReadVariableOp_30ReadVariableOptraining/Adam/dense_5/bias/v^AssignVariableOp_30*
dtype0*
_output_shapes
:
Y
save/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 

save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_9590bf5c7cb241b1a82342ca39c7df01/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
ж
save/SaveV2/tensor_namesConst*
dtype0*
valueџBќ#Bdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBdense_4/biasBdense_4/kernelBdense_5/biasBdense_5/kernelBtraining/Adam/beta_1Btraining/Adam/beta_2Btraining/Adam/decayBtraining/Adam/dense_1/bias/mBtraining/Adam/dense_1/bias/vBtraining/Adam/dense_1/kernel/mBtraining/Adam/dense_1/kernel/vBtraining/Adam/dense_2/bias/mBtraining/Adam/dense_2/bias/vBtraining/Adam/dense_2/kernel/mBtraining/Adam/dense_2/kernel/vBtraining/Adam/dense_3/bias/mBtraining/Adam/dense_3/bias/vBtraining/Adam/dense_3/kernel/mBtraining/Adam/dense_3/kernel/vBtraining/Adam/dense_4/bias/mBtraining/Adam/dense_4/bias/vBtraining/Adam/dense_4/kernel/mBtraining/Adam/dense_4/kernel/vBtraining/Adam/dense_5/bias/mBtraining/Adam/dense_5/bias/vBtraining/Adam/dense_5/kernel/mBtraining/Adam/dense_5/kernel/vBtraining/Adam/iterBtraining/Adam/learning_rate*
_output_shapes
:#
Љ
save/SaveV2/shape_and_slicesConst*
_output_shapes
:#*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ц
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp0training/Adam/dense_1/bias/m/Read/ReadVariableOp0training/Adam/dense_1/bias/v/Read/ReadVariableOp2training/Adam/dense_1/kernel/m/Read/ReadVariableOp2training/Adam/dense_1/kernel/v/Read/ReadVariableOp0training/Adam/dense_2/bias/m/Read/ReadVariableOp0training/Adam/dense_2/bias/v/Read/ReadVariableOp2training/Adam/dense_2/kernel/m/Read/ReadVariableOp2training/Adam/dense_2/kernel/v/Read/ReadVariableOp0training/Adam/dense_3/bias/m/Read/ReadVariableOp0training/Adam/dense_3/bias/v/Read/ReadVariableOp2training/Adam/dense_3/kernel/m/Read/ReadVariableOp2training/Adam/dense_3/kernel/v/Read/ReadVariableOp0training/Adam/dense_4/bias/m/Read/ReadVariableOp0training/Adam/dense_4/bias/v/Read/ReadVariableOp2training/Adam/dense_4/kernel/m/Read/ReadVariableOp2training/Adam/dense_4/kernel/v/Read/ReadVariableOp0training/Adam/dense_5/bias/m/Read/ReadVariableOp0training/Adam/dense_5/bias/v/Read/ReadVariableOp2training/Adam/dense_5/kernel/m/Read/ReadVariableOp2training/Adam/dense_5/kernel/v/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOp*1
dtypes'
%2#	

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
T0*

axis *
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
й
save/RestoreV2/tensor_namesConst*
valueџBќ#Bdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBdense_4/biasBdense_4/kernelBdense_5/biasBdense_5/kernelBtraining/Adam/beta_1Btraining/Adam/beta_2Btraining/Adam/decayBtraining/Adam/dense_1/bias/mBtraining/Adam/dense_1/bias/vBtraining/Adam/dense_1/kernel/mBtraining/Adam/dense_1/kernel/vBtraining/Adam/dense_2/bias/mBtraining/Adam/dense_2/bias/vBtraining/Adam/dense_2/kernel/mBtraining/Adam/dense_2/kernel/vBtraining/Adam/dense_3/bias/mBtraining/Adam/dense_3/bias/vBtraining/Adam/dense_3/kernel/mBtraining/Adam/dense_3/kernel/vBtraining/Adam/dense_4/bias/mBtraining/Adam/dense_4/bias/vBtraining/Adam/dense_4/kernel/mBtraining/Adam/dense_4/kernel/vBtraining/Adam/dense_5/bias/mBtraining/Adam/dense_5/bias/vBtraining/Adam/dense_5/kernel/mBtraining/Adam/dense_5/kernel/vBtraining/Adam/iterBtraining/Adam/learning_rate*
_output_shapes
:#*
dtype0
Ќ
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:#*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Н
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*Ђ
_output_shapes
:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	
N
save/Identity_1Identitysave/RestoreV2*
_output_shapes
:*
T0
U
save/AssignVariableOpAssignVariableOpdense_1/biassave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
Y
save/AssignVariableOp_1AssignVariableOpdense_1/kernelsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
W
save/AssignVariableOp_2AssignVariableOpdense_2/biassave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
_output_shapes
:*
T0
Y
save/AssignVariableOp_3AssignVariableOpdense_2/kernelsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
_output_shapes
:*
T0
W
save/AssignVariableOp_4AssignVariableOpdense_3/biassave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
_output_shapes
:*
T0
Y
save/AssignVariableOp_5AssignVariableOpdense_3/kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
_output_shapes
:*
T0
W
save/AssignVariableOp_6AssignVariableOpdense_4/biassave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
_output_shapes
:*
T0
Y
save/AssignVariableOp_7AssignVariableOpdense_4/kernelsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
_output_shapes
:*
T0
W
save/AssignVariableOp_8AssignVariableOpdense_5/biassave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
Z
save/AssignVariableOp_9AssignVariableOpdense_5/kernelsave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
_output_shapes
:*
T0
a
save/AssignVariableOp_10AssignVariableOptraining/Adam/beta_1save/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
_output_shapes
:*
T0
a
save/AssignVariableOp_11AssignVariableOptraining/Adam/beta_2save/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
_output_shapes
:*
T0
`
save/AssignVariableOp_12AssignVariableOptraining/Adam/decaysave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
_output_shapes
:*
T0
i
save/AssignVariableOp_13AssignVariableOptraining/Adam/dense_1/bias/msave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
_output_shapes
:*
T0
i
save/AssignVariableOp_14AssignVariableOptraining/Adam/dense_1/bias/vsave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
k
save/AssignVariableOp_15AssignVariableOptraining/Adam/dense_1/kernel/msave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
_output_shapes
:*
T0
k
save/AssignVariableOp_16AssignVariableOptraining/Adam/dense_1/kernel/vsave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
_output_shapes
:*
T0
i
save/AssignVariableOp_17AssignVariableOptraining/Adam/dense_2/bias/msave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
i
save/AssignVariableOp_18AssignVariableOptraining/Adam/dense_2/bias/vsave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
_output_shapes
:*
T0
k
save/AssignVariableOp_19AssignVariableOptraining/Adam/dense_2/kernel/msave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
_output_shapes
:*
T0
k
save/AssignVariableOp_20AssignVariableOptraining/Adam/dense_2/kernel/vsave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
_output_shapes
:*
T0
i
save/AssignVariableOp_21AssignVariableOptraining/Adam/dense_3/bias/msave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
_output_shapes
:*
T0
i
save/AssignVariableOp_22AssignVariableOptraining/Adam/dense_3/bias/vsave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
_output_shapes
:*
T0
k
save/AssignVariableOp_23AssignVariableOptraining/Adam/dense_3/kernel/msave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
k
save/AssignVariableOp_24AssignVariableOptraining/Adam/dense_3/kernel/vsave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
_output_shapes
:*
T0
i
save/AssignVariableOp_25AssignVariableOptraining/Adam/dense_4/bias/msave/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
_output_shapes
:*
T0
i
save/AssignVariableOp_26AssignVariableOptraining/Adam/dense_4/bias/vsave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
_output_shapes
:*
T0
k
save/AssignVariableOp_27AssignVariableOptraining/Adam/dense_4/kernel/msave/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
T0*
_output_shapes
:
k
save/AssignVariableOp_28AssignVariableOptraining/Adam/dense_4/kernel/vsave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
_output_shapes
:*
T0
i
save/AssignVariableOp_29AssignVariableOptraining/Adam/dense_5/bias/msave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
i
save/AssignVariableOp_30AssignVariableOptraining/Adam/dense_5/bias/vsave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
_output_shapes
:*
T0
k
save/AssignVariableOp_31AssignVariableOptraining/Adam/dense_5/kernel/msave/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
k
save/AssignVariableOp_32AssignVariableOptraining/Adam/dense_5/kernel/vsave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
_output_shapes
:*
T0	
_
save/AssignVariableOp_33AssignVariableOptraining/Adam/itersave/Identity_34*
dtype0	
R
save/Identity_35Identitysave/RestoreV2:34*
T0*
_output_shapes
:
h
save/AssignVariableOp_34AssignVariableOptraining/Adam/learning_ratesave/Identity_35*
dtype0
П
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"я+
	variablesс+о+
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

dense_4/kernel:0dense_4/kernel/Assign$dense_4/kernel/Read/ReadVariableOp:0(2+dense_4/kernel/Initializer/random_uniform:08
o
dense_4/bias:0dense_4/bias/Assign"dense_4/bias/Read/ReadVariableOp:0(2 dense_4/bias/Initializer/zeros:08

dense_5/kernel:0dense_5/kernel/Assign$dense_5/kernel/Read/ReadVariableOp:0(2+dense_5/kernel/Initializer/random_uniform:08
o
dense_5/bias:0dense_5/bias/Assign"dense_5/bias/Read/ReadVariableOp:0(2 dense_5/bias/Initializer/zeros:08
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
Е
 training/Adam/dense_4/kernel/m:0%training/Adam/dense_4/kernel/m/Assign4training/Adam/dense_4/kernel/m/Read/ReadVariableOp:0(22training/Adam/dense_4/kernel/m/Initializer/zeros:0
­
training/Adam/dense_4/bias/m:0#training/Adam/dense_4/bias/m/Assign2training/Adam/dense_4/bias/m/Read/ReadVariableOp:0(20training/Adam/dense_4/bias/m/Initializer/zeros:0
Е
 training/Adam/dense_5/kernel/m:0%training/Adam/dense_5/kernel/m/Assign4training/Adam/dense_5/kernel/m/Read/ReadVariableOp:0(22training/Adam/dense_5/kernel/m/Initializer/zeros:0
­
training/Adam/dense_5/bias/m:0#training/Adam/dense_5/bias/m/Assign2training/Adam/dense_5/bias/m/Read/ReadVariableOp:0(20training/Adam/dense_5/bias/m/Initializer/zeros:0
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
training/Adam/dense_3/bias/v:0#training/Adam/dense_3/bias/v/Assign2training/Adam/dense_3/bias/v/Read/ReadVariableOp:0(20training/Adam/dense_3/bias/v/Initializer/zeros:0
Е
 training/Adam/dense_4/kernel/v:0%training/Adam/dense_4/kernel/v/Assign4training/Adam/dense_4/kernel/v/Read/ReadVariableOp:0(22training/Adam/dense_4/kernel/v/Initializer/zeros:0
­
training/Adam/dense_4/bias/v:0#training/Adam/dense_4/bias/v/Assign2training/Adam/dense_4/bias/v/Read/ReadVariableOp:0(20training/Adam/dense_4/bias/v/Initializer/zeros:0
Е
 training/Adam/dense_5/kernel/v:0%training/Adam/dense_5/kernel/v/Assign4training/Adam/dense_5/kernel/v/Read/ReadVariableOp:0(22training/Adam/dense_5/kernel/v/Initializer/zeros:0
­
training/Adam/dense_5/bias/v:0#training/Adam/dense_5/bias/v/Assign2training/Adam/dense_5/bias/v/Read/ReadVariableOp:0(20training/Adam/dense_5/bias/v/Initializer/zeros:0"п	
trainable_variablesЧ	Ф	
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

dense_4/kernel:0dense_4/kernel/Assign$dense_4/kernel/Read/ReadVariableOp:0(2+dense_4/kernel/Initializer/random_uniform:08
o
dense_4/bias:0dense_4/bias/Assign"dense_4/bias/Read/ReadVariableOp:0(2 dense_4/bias/Initializer/zeros:08

dense_5/kernel:0dense_5/kernel/Assign$dense_5/kernel/Read/ReadVariableOp:0(2+dense_5/kernel/Initializer/random_uniform:08
o
dense_5/bias:0dense_5/bias/Assign"dense_5/bias/Read/ReadVariableOp:0(2 dense_5/bias/Initializer/zeros:08*Њ
serving_default
9
input_image*
	input_1:0џџџџџџџџџрр=
dense_5/Softmax:0(
dense_5/Softmax:0џџџџџџџџџtensorflow/serving/predict