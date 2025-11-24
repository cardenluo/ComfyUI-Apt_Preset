
from copy import deepcopy
import logging
import numpy as np
import torch
import math
from numbers import Number
from abc import ABC, abstractmethod
from functools import reduce, partial
import operator
from sortedcontainers import SortedDict
from typing import Tuple, Optional, Union, Dict, Callable
from collections import UserDict
import random, string
from ..main_unit import *





try:
    from sortedcontainers import SortedDict
except ImportError:
    class SortedDict(dict):
        def keys(self):
            return sorted(super().keys())

        def items(self):
            return sorted(super().items(), key=lambda x: x[0])

        def peekitem(self, index):
            keys = sorted(self.keys())
            if 0 <= index < len(keys):
                key = keys[index]
                return (key, self[key])
            raise IndexError("index out of range for peekitem")





#region-------------------------依赖--------------------

#--------------------------------------utils.py---------------------------------
def simplify(curve):
    j=1
    while j < (len(curve._data)-1):
        kf_prev = curve._data.peekitem(j-1)[1]
        kf_this = curve._data.peekitem(j)[1]
        kf_next = curve._data.peekitem(j+1)[1]
        
        if not (kf_prev.value == kf_this.value == kf_next.value):
            j+=1
            continue
        if not (kf_prev.interpolation_method == kf_this.interpolation_method == kf_next.interpolation_method):
            j+=1
            continue
        if not (kf_prev._interpolator_arguments == kf_this._interpolator_arguments == kf_next._interpolator_arguments):
            j+=1
            continue
        curve._data.popitem(j)
    return curve


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):

    return ''.join(random.choice(chars) for _ in range(size))


class DictValuesArithmeticFriendly(UserDict):
    def __arithmetic_helper(self, operator, other=None):
        outv = deepcopy(self)
        for k,v in self.items():
            if other is not None:
                outv[k] = operator(v, other)
            else:
                outv[k] = operator(v)
        return outv
    def __add__(self, other):
        return self.__arithmetic_helper(operator.add, other)
    #def __div__(self, other):
    def __truediv__(self, other): # oh right
        return self.__arithmetic_helper(operator.truediv, other)
        #return self.__arithmetic_helper(1/other, operator.mul)
    def __rtruediv__(self, other):
        outv = deepcopy(self)
        for k,v in self.items():
                outv[k] = other / v
        return outv
    def __mul__(self, other):
        return self.__arithmetic_helper(operator.mul, other)
    def __neg__(self):
        return self.__arithmetic_helper(operator.neg)
    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other
    def __rsub__(self, other):
        return (self * (-1)) + other
    def __sub__(self, other):
        return self.__arithmetic_helper(operator.sub, other)











#----------------------------------------------------interpolation.py

def bisect_left_keyframe(k: Number, curve:'Curve', *args, **kargs) -> 'Keyframe':
    """
    finds the value of the keyframe in a sorted dictionary to the left of a given key, i.e. performs "previous" interpolation
    """
    self=curve
    right_index = self._data.bisect_right(k)
    left_index = right_index - 1
    if right_index > 0:
        _, left_value = self._data.peekitem(left_index)
    else:
        raise RuntimeError(
            "The return value of bisect_right should always be greater than zero, "
            f"however self._data.bisect_right({k}) returned {right_index}."
            "You should never see this error. Please report the circumstances to the library issue tracker on github."
            )
    return left_value

def bisect_left_value(k: Number, curve:'Curve', *args, **kargs) -> 'Keyframe':
    kf = bisect_left_keyframe(k, curve)
    return kf.value

def bisect_right_keyframe(k:Number, curve:'Curve', *args, **kargs) -> 'Keyframe':
    """
    finds the value of the keyframe in a sorted dictionary to the right of a given key, i.e. performs "next" interpolation
    """
    self=curve
    right_index = self._data.bisect_right(k)
    if right_index > 0:
        _, right_value = self._data.peekitem(right_index)
    else:
        raise RuntimeError(
            "The return value of bisect_right should always be greater than zero, "
            f"however self._data.bisect_right({k}) returned {right_index}."
            "You should never see this error. Please report the circumstances to the library issue tracker on github."
            )
    return right_value

def bisect_right_value(k: Number, curve:'Curve', *args, **kargs) -> 'Keyframe':
    kf = bisect_right_keyframe(k, curve)
    return kf.value

def sin2(t:Number) -> Number:
    # Suggestion and formula courtesy of Katherine Crowson
    return (math.sin(t * math.pi / 2)) ** 2

# to do: turn this into a decorator in dmarx/Keyframed
def eased_lerp(k:Number, curve:'Curve', ease:Callable=sin2, *args, **kargs) -> Number:
    left = bisect_left_keyframe(k, curve)
    right = bisect_right_keyframe(k, curve)
    xs = [left.t, right.t]
    ys = [left.value, right.value]

    span = xs[1]-xs[0]
    t = (k-xs[0]) / span
    t_new = ease(t)
    return ys[1] * t_new + ys[0] * (1-t_new)

def linear(k, curve, *args, **kargs):
    left = bisect_left_keyframe(k, curve)
    right = bisect_right_keyframe(k, curve)
    x0, x1 = left.t, right.t
    y0, y1 = left.value, right.value



    d = x1-x0
    t = (x1-k)/d
    outv =  t*y0 + (1-t)*y1
    return outv

def exp_decay(t, curve, decay_rate):
    kf_prev = bisect_left_keyframe(t, curve)
    td = max(t- kf_prev.t, 0)
    v0 = kf_prev.value
    return v0 * math.exp(-td * decay_rate)

def sine_wave(t, curve, wavelength=None, frequency=None, phase=0, amplitude=1):
    if (wavelength is None): 
        if (frequency is not None):
            wavelength = 1/frequency
        else:
            wavelength = 4 # interpolate from 0 to pi/2
    return amplitude * math.sin(2*math.pi*t / wavelength + phase)

INTERPOLATORS={
    None:bisect_left_value,
    'previous':bisect_left_value,
    'next':bisect_right_value,
    'eased_lerp':eased_lerp,
    'linear':linear,
    'exp_decay':exp_decay,
    'sine_wave':sine_wave,
}

EASINGS={
    None:bisect_left_value,
    'previous':bisect_left_value,
    'next':bisect_right_value,
    'linear':partial(eased_lerp, ease=lambda t: t),
    'sin':partial(eased_lerp, ease=lambda t: math.sin(t * math.pi / 2)),
    'sin^2':partial(eased_lerp, ease=lambda t: math.sin(t * math.pi / 2)**2),

}


def register_interpolation_method(name:str, f:Callable):
    """
    Adds a new interpolation method to the INTERPOLATORS registry.
    """
    INTERPOLATORS[name] = f

def get_context_left(k, curve, n, eps=1e-9):
  kfs = []
  while len(kfs) < n:
    k = bisect_left_keyframe(k, curve).t
    kfs.append(k)
    if k == 0:
      break
    k -= eps
  return kfs

def get_context_right(k, curve, n, eps=1e-9):
  kfs = []
  while len(kfs) < n:
    k = bisect_right_keyframe(k, curve).t
    kfs.append(k)
    if k == 0:
      break
    k += eps
  return kfs



















#-----------------------------------------------------------------------curve.py
def is_torch_tensor(obj):
    try:
        import torch
        return isinstance(obj, torch.Tensor)
    except ImportError:
        pass
    return False

def is_numpy_ndarray(obj):
    try:
        import numpy as np
        return isinstance(obj, np.ndarray)
    except ImportError:
        pass
    return False

def numpy_array_equal(a,b):
    import numpy as np
    return np.array_equal(a,b)

def torch_isequal(a,b):
    import torch
    return torch.equal(a,b)

# workhorse of Curve.__init__, should probably attach it as an instance method on Curve
def ensure_sorteddict_of_keyframes(
    curve: 'Curve',
    default_interpolation:Union[str,Callable]='previous',
    default_interpolator_args = None,
) -> SortedDict:

    if isinstance(curve, SortedDict):
        sorteddict = curve
    elif isinstance(curve, dict):
        sorteddict = SortedDict(curve)
    #elif isinstance(curve, (Number, np.ndarray, torch.Tensor)):
    elif (isinstance(curve, Number) or is_numpy_ndarray(curve) or is_torch_tensor(curve)):
        sorteddict = SortedDict({0:Keyframe(t=0,value=curve, interpolation_method=default_interpolation, interpolator_arguments=default_interpolator_args)})
    elif (isinstance(curve, list) or isinstance(curve, tuple)):
        d_ = {}
        # aaaand here we go again.
        implied_interpolation = default_interpolation
        implied_interpolator_args = default_interpolator_args
        for item in curve:
            if not isinstance(item, Keyframe):
                if len(item) == 2:
                    item = (item[0], item[1], implied_interpolation, implied_interpolator_args)
                elif len(item) == 3:
                    item = (item[0], item[1], item[2], implied_interpolator_args)
                item = Keyframe(*item)
            implied_interpolation = item.interpolation_method
            implied_interpolator_args = item.interpolator_arguments
            d_[item.t] = item
        sorteddict = SortedDict(d_)
    else:
        raise NotImplementedError

    d_ = {}
    implied_interpolation = default_interpolation
    implied_interpolator_args = default_interpolator_args
    if 0 not in sorteddict:
        d_[0] = Keyframe(t=0,value=0, interpolation_method=implied_interpolation, interpolator_arguments=implied_interpolator_args)
    for k,v in sorteddict.items():
        if isinstance(v, Keyframe):
            implied_interpolation = v.interpolation_method
            implied_interpolator_args = v.interpolator_arguments
            d_[k] = v
        elif isinstance(v, dict):
            kf = Keyframe(**v)
            if 'interpolation_method' not in v:
                kf.interpolation_method = implied_interpolation
                kf.interpolator_arguments = implied_interpolator_args
            implied_interpolation = kf.interpolation_method
            implied_interpolator_args = kf.interpolator_arguments
            if k != kf.t:
                kf.t = k
            d_[k] = kf
        elif isinstance(v, list) or isinstance(v, tuple):
            kf = Keyframe(*v)
            if len(v) < 3:
                kf.interpolation_method = implied_interpolation
                kf.interpolator_arguments = implied_interpolator_args
            implied_interpolation = kf.interpolation_method
            implied_interpolator_args = kf.interpolator_arguments
            d_[k] = kf
        #elif isinstance(v, (Number, np.ndarray, torch.Tensor)):
        elif (isinstance(v, Number) or is_numpy_ndarray(v) or is_torch_tensor(v)):
            d_[k] = Keyframe(t=k,value=v, interpolation_method=implied_interpolation, interpolator_arguments=implied_interpolator_args)
        else:
            raise NotImplementedError
    return SortedDict(d_)


class Keyframe:

    def __init__(
        self,
        t:Number,
        value,
        interpolation_method:Optional[Union[str,Callable]]=None,
        interpolator_arguments=None,
        label=None,
    ):
        self.t=t
        self.label = label
        #self.value=value
        ### <chatgpt>
        #if isinstance(value, np.ndarray):
        if is_numpy_ndarray(value):
            #self.value = np.array(value)  # Ensure a copy of the array is stored
            self.value = deepcopy(value)
        #elif isinstance(value, torch.Tensor):
        elif is_torch_tensor(value):
            self.value = value.clone()    # Ensure a clone of the tensor is stored
        else:
            self.value = value
        ### </chatgpt>
        self.interpolation_method=interpolation_method
        if interpolator_arguments is None:
            interpolator_arguments = {}
        self._interpolator_arguments = interpolator_arguments
    
    @property
    def interpolator_arguments(self):
        if hasattr(self, '_interpolator_arguments'):
            return self._interpolator_arguments
        return {}

    def __eq__(self, other) -> bool:

        if is_numpy_ndarray(self.value):
            return numpy_array_equal(self.value, other)
        elif is_torch_tensor(self.value):
            return torch_isequal(self.value, other)
        else:
            return self.value == other
    def __repr__(self) -> str:
        #d = f"Keyframe(t={self.t}, value={self.value}, interpolation_method='{self.interpolation_method}')"
        d = self.to_dict()
        return f"Keyframe({d})"
    def _to_dict(self, *args, **kwargs) -> dict:
        d = {'t':self.t, 'value':self.value, 'interpolation_method':self.interpolation_method}
        if self.interpolator_arguments:
            d['interpolator_arguments'] = self.interpolator_arguments
        if self.label is not None:
            d['label'] = self.label

        if is_numpy_ndarray(self.value):
            d['value'] = self.value.tolist()
        #elif isinstance(self.value, torch.Tensor):
        elif is_torch_tensor(self.value):
            d['value'] = self.value.numpy().tolist()
        else:
            d['value'] = self.value
        ### </chatgpt>
        return d
    def _to_tuple(self, *args, **kwags):
        if not self.interpolator_arguments:
            return (self.t, self.value, self.interpolation_method)
        return (self.t, self.value, self.interpolation_method, self.interpolator_arguments)
    def to_dict(self, *args, **kwags):
        return self._to_dict(*args, **kwags)
        #return self._to_tuple(*args, **kwags)

class CurveBase(ABC):
    def copy(self) -> 'CurveBase':
        return deepcopy(self)

    @property
    @abstractmethod
    def keyframes(self) -> list:
        pass

    @property
    @abstractmethod
    def values(self) -> list:
        pass
    
    @property
    @abstractmethod
    def duration(self) -> Number:
        pass 

    @abstractmethod
    def __getitem__(self, k) -> Number:
        pass

    def _adjust_k_for_looping(self, k:Number) -> Number:
        n = (self.duration + 1)
        if self.loop and k >= max(self.keyframes):
            k %= n
        elif self.bounce:
            n2 = 2*(n-1)
            k %= n2
            if k >= n:
                k = n2 - k
        return k

    def plot(self, n:int=None, xs:list=None, eps:float=1e-9, *args, **kargs):

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Please install matplotlib to use Curve.plot()")
        if xs is None:
            if n is None:
                n = self.duration + 1
            xs_base = list(range(int(n))) + list(self.keyframes)
            xs = set()
            for x in xs_base:
                if (x>0) and (eps is not None) and (eps > 0):
                    xs.add(x-eps)
                xs.add(x)
        xs = list(set(xs))
        xs.sort()
        
        ys = [self[x] for x in xs]
        if kargs.get('label') is None:
            kargs['label']=self.label
        line = plt.plot(xs, ys, *args, **kargs)
        kfx = self.keyframes
        kfy = [self[x] for x in kfx]
        plt.scatter(kfx, kfy, color=line[0].get_color())

    def random_label(self) -> str:
        return f"curve_{id_generator()}"

    def __sub__(self, other) -> 'CurveBase':
        return self + (-1 * other)

    def __rsub__(self, other) -> 'CurveBase':
        return (-1*self) + other

    def __radd__(self,other) -> 'CurveBase':
        return self+other

    def __rmul__(self, other) -> 'CurveBase':
        return self*other

    def __neg__(self) -> 'CurveBase':
        return self * (-1)

    def __eq__(self, other) -> bool:
        return self.to_dict(simplify=True, ignore_labels=True) == other.to_dict(simplify=True, ignore_labels=True)
    @abstractmethod
    def to_dict(simplify=False, for_yaml=False, ignore_labels=False):
        raise NotImplementedError


class Curve(CurveBase):

    def __init__(self,
        curve: Union[
            int,
            float,
            Dict,
            SortedDict,
            Tuple[Tuple[Number, Number]],
        ] = ((0,0),),
        default_interpolation='previous',
        default_interpolator_args=None,
        loop: bool = False,
        bounce: bool = False,
        duration:Optional[float]=None,
        label:str=None,
    ):

        if isinstance(curve, type(self)):
            self._data = curve._data
        else:
            self._data = ensure_sorteddict_of_keyframes(
                curve,
                default_interpolation=default_interpolation,
                default_interpolator_args=default_interpolator_args,
            )

        #self.default_interpolation=default_interpolation # to do: this doesn't need to be a Curve attribute
        self.loop=loop
        self.bounce=bounce
        self._duration=duration
        if label is None:
            label = self.random_label()
            self._using_default_label = True
        self.label=str(label)

    @property
    def keyframes(self) -> list:
        return self._data.keys()
    
    @property
    def values(self) -> list:
        # not a fan of this
        return [kf.value for kf in self._data.values()]

    @property
    def duration(self) -> Number:
        if self._duration:
            return self._duration
        return max(self.keyframes)

    def __get_slice(self, k:slice):
        start, end = k.start, k.stop
        if (start is None) and (end is None):
            return self.copy()
        if start is None:
            start = 0
        elif start < 0:
            start = self.keyframes[start]
        if end is None:
            end = self.duration
        elif end < 0:
            end = self.keyframes[end]
        d = {}
        for k, kf in self._data.items():
            if start <= k <= end:
                d[k] = deepcopy(kf)
        for k in (start, end):
            if (k is not None) and (k not in d):
                #interp = bisect_left_keyframe(k, self).interpolation_method
                kf0 = bisect_left_keyframe(k, self)
                interp_args = kf0.interpolator_arguments
                kf = Keyframe(
                    t=k,
                    value=self[k],
                    interpolation_method=kf0.interpolation_method,
                    interpolator_arguments=interp_args if interp_args else None,
                )
                d[k] = kf
        # reindex to slice origin
        #d = {(k-start):kf for k,kf in d.items()}
        d2 = {}
        for k,kf in d.items():
            k_shifted = k-start
            kf.t = k_shifted
            d2[k_shifted] = kf
        d = d2

        #loop = self.loop if end# to do: revisit the logic here
        loop = False # let's just keep it like this for simplicity. if someone wants a slice output to loop, they can be explicit
        bounce = False
        return Curve(curve=d, loop=loop, bounce=bounce, duration=end)

    def __getitem__(self, k:Number) -> Number:

        if isinstance(k, slice):
            return self.__get_slice(k)

        k = self._adjust_k_for_looping(k)

        if k in self._data.keys():
            outv = self._data[k]
            if isinstance(outv, Keyframe):
                outv = outv.value
            return outv

        left_value = bisect_left_keyframe(k, self)
        interp = left_value.interpolation_method

        if (interp is None) or isinstance(interp, str):
            f = EASINGS.get(interp)
            if f is None:
                f = INTERPOLATORS.get(interp)
            if f is None:
                raise ValueError(f"Unsupported interpolation method: {interp}")
        elif isinstance(interp, Callable):
            f = interp
        else:
            raise ValueError(f"Unsupported interpolation method: {interp}")
        
        interp_args = left_value.interpolator_arguments
        if interp_args:
            f = partial(f, **interp_args)
        
        try:
            return f(k, self)
        except IndexError:
            return left_value.value
    
    def __setitem__(self, k, v):
        interp_args = None
        if not isinstance(v, Keyframe):
            if isinstance(v, Callable):
                interp = v
                v = self[k]
            else:
                kf = bisect_left_keyframe(k,self)
                interp = kf.interpolation_method
                interp_args = kf.interpolator_arguments
            v = Keyframe(
                t=k,
                value=v,
                interpolation_method=interp,
                interpolator_arguments=interp_args if interp_args else None,
            )
        self._data[k] = v
    
    def __str__(self) -> str:
        d_ = {k:self[k] for k in self.keyframes}
        return f"Curve({d_}"

    def __add__(self, other) -> CurveBase:
        if not isinstance(other, CurveBase):
            other = Curve(other)
        return self.__add_curves__(other)

    def __add_curves__(self, other) -> 'Composition':
        if isinstance(other, ParameterGroup):
            # this triggers the operator to get resolved by "other" instead of self
            return NotImplemented
        params = {self.label:self, other.label:other}
        new_label = '+'.join(params.keys())
        return Composition(parameters=params, label=new_label, reduction='add')

    def __mul__(self, other) -> CurveBase:
        if isinstance(other, CurveBase):
            return self.__mul_curves__(other)
        label_ = f"( {other} * {self.label} )"
        other = Curve(other)
        return Composition(parameters=(self, other), label=label_, reduction='multiply')
    
    def __mul_curves__(self, other) -> 'Composition':
        if isinstance(other, ParameterGroup):
            # this triggers the operator to get resolved by "other" instead of self
            return NotImplemented
        params = {self.label:self, other.label:other}
        new_label = '*'.join(params.keys())
        #params = ParameterGroup(params) ## added... no difference
        return Composition(parameters=params, label=new_label, reduction='multiply')

    @classmethod
    def from_function(cls, f:Callable) -> CurveBase:
        return cls({0:f(0)}, default_interpolation=lambda k, _: f(k))

    def to_dict(self, simplify=False, for_yaml=False, ignore_labels=False):

        if for_yaml:
            d_curve = tuple([kf._to_tuple(simplify=simplify) for k, kf in self._data.items()])
        else:
            d_curve = {k:kf.to_dict(simplify=simplify) for k, kf in self._data.items()}
        
        # to do: make this less ugly
        if simplify:
            d_curve = {}
            recs = []
            #for t, v, kf_interp in 
            implied_interpolation = 'previous'
            implied_interpolator_arguments = {}
            for kf in self._data.values():
                if ((kf.t == 0) and (kf.value == 0) and (kf.interpolation_method == implied_interpolation)):
                    continue
                rec = {'t':kf.t,'value':kf.value}
                if kf.interpolation_method != implied_interpolation:
                    rec['interpolation_method'] = kf.interpolation_method
                    implied_interpolation = kf.interpolation_method

                if kf.interpolator_arguments != implied_interpolator_arguments:
                    rec['interpolator_arguments'] = kf.interpolator_arguments
                    implied_interpolator_arguments = kf.interpolator_arguments
                    

                if for_yaml:
                    rec = tuple(rec.values())
                    recs.append(rec)
                else:
                    t = rec.pop('t')
                    d_curve[t] = rec

            if for_yaml:
                outv = {'curve': recs}
            else:
                outv = {'curve':d_curve}

            if self.duration != kf.t: # test against timestamp of last scene keyframe 
                outv['duration'] = self.duration
            if self.loop:
                outv['loop'] = self.loop
            if self.bounce:
                outv['bounce'] = self.bounce
            # uh... ignore default labels I guess? Maybe make that an option?
            if not (hasattr(self, '_using_default_label') and self.label.startswith('curve_')):
                outv['label'] = self.label
            
        else:
            outv = dict(
            curve=d_curve,
            loop=self.loop,
            bounce=self.bounce,
            duration=self.duration,
            label=self.label,
        )

        if ignore_labels and 'label' in outv:
            outv.pop('label')

        return outv
    
    def append(self, other):
        if not isinstance(other, CurveBase):
            raise NotImplementedError
        if not isinstance(other, Curve):
            return NotImplemented # delegate figuring out what to do to the other object
        delta = self.duration + 1
        for t0, kf in other.copy()._data.items():
            t = delta + t0
            kf.t = t
            self._data[t] = kf
        return self


class ParameterGroup(CurveBase):

    def __init__(
        self,
        parameters:Union[Dict[str, Curve],'ParameterGroup', list, tuple],
        weight:Optional[Union[Curve,Number]]=1,
        label=None,
        loop: bool = False,
        bounce: bool = False,
    ):
        self.loop, self.bounce = loop, bounce
        if isinstance(parameters, list) or isinstance(parameters, tuple):
            d = {}
            for curve in parameters:
                if not isinstance(curve, CurveBase):
                    curve = Curve(curve)
                d[curve.label] = curve
            parameters = d

        if isinstance(parameters, ParameterGroup):
            pg = parameters
            self.parameters = pg.parameters
            self._weight = pg.weight
            if label is None:
                label = pg.label # to do: I think this should probably be a random label gen
            self.label = str(label)
            return
        self.parameters = {}
        for name, v in parameters.items():
            if not isinstance(v, CurveBase):
                v = Curve(v)
            v.label = str(name)
            self.parameters[name] = v
        if label is None:
            label = self.random_label()
            self._using_default_label = True
        self.label = str(label)
        if not isinstance(weight, Curve):
            weight = Curve(weight)
            weight._using_default_label = True
        self._weight = weight
    
    @property
    def weight(self):
        # defining this as a property so we can override the label to 
        # always match the label of the associated ParameterGroup
        self._weight.label = f"{self.label}_WEIGHT"
        self._weight._using_default_label = True
        return self._weight

    def __get_slice(self, k) -> 'ParameterGroup':
        outv = self.copy()
        outv.parameters = {name:param[k] for name, param in self.parameters.items()}
        outv._weight = outv.weight[k]
        return outv

    def __getitem__(self, k) -> dict:
        if isinstance(k, slice):
            return self.__get_slice(k)
        k = self._adjust_k_for_looping(k)
        wt = self.weight[k]
        d = {name:param[k]*wt for name, param in self.parameters.items() }
        return DictValuesArithmeticFriendly(d)

    def copy(self) -> 'ParameterGroup':
        return deepcopy(self)

    # feels a bit redundant with DictValuesArithmeticFriendly, but fuck it.
    def __add__(self, other) -> 'ParameterGroup':
        outv = self.copy()
        for k,v in outv.parameters.items():
            outv.parameters[k] = v + other
        return outv
    
    def __mul__(self, other) -> 'ParameterGroup':
        outv = self.copy()
        for k,v in outv.parameters.items():
            outv.parameters[k] = v * other
        return outv
    
    def __truediv__(self, other) -> 'ParameterGroup':
        outv = self.copy()
        for k,v in outv.parameters.items():
            outv.parameters[k] = v / other
        return outv
    
    def __rtruediv__(self, other) -> 'ParameterGroup':
        outv = self.copy()
        for k,v in outv.parameters.items():
            outv.parameters[k] = other / v
        return outv

    def __eq__(self, other) -> bool:
        return self.to_dict(simplify=True, ignore_labels=True)['parameters'] == other.to_dict(simplify=True, ignore_labels=True)['parameters']

    @property
    def duration(self) -> Number:
        return max(curve.duration for curve in self.parameters.values())

    def plot(self, n:int=None, xs:list=None, eps:float=1e-9, *args, **kargs):
        if n is None:
            n = self.duration + 1
        for curve in self.parameters.values():
            curve = curve.copy()
            curve = curve * self.weight
            curve.plot(n=n, xs=xs, eps=eps, *args, **kargs)

    @property
    def keyframes(self) -> list:
        kfs = set()
        for curve in self.parameters.values():
            kfs.update(curve.keyframes)
        kfs = list(kfs) 
        kfs.sort()
        return kfs

    @property
    def values(self) -> list:
        return [self[k] for k in self.keyframes]

    def random_label(self) -> str:
        return f"pgroup({','.join([c.label for c in self.parameters.values()])})"
    def to_dict(self, simplify=False, for_yaml=False, ignore_labels=False):
        params = {k:v.to_dict(simplify=simplify, for_yaml=for_yaml, ignore_labels=ignore_labels) for k,v in self.parameters.items()}
        weight = self.weight.to_dict(simplify=simplify, for_yaml=for_yaml, ignore_labels=ignore_labels)
        
        if not simplify:
            outv= dict(
                parameters=params,
                weight=weight,
                #label=self.label,
            )
            if not ignore_labels:
                outv['label'] = self.label
            return outv
        else:
            for k in list(params.keys()):
                if 'label' in params[k]:
                    params[k].pop('label')

            outv = {'parameters':params}
            wt2 = deepcopy(weight)
            if 'label' in wt2:
                wt2.pop('label')
            if wt2 != Curve(1).to_dict(simplify=simplify, for_yaml=for_yaml, ignore_labels=ignore_labels):
                outv['weight'] = wt2 #weight
            if not hasattr(self, '_using_default_label') and not ignore_labels:
                outv['label'] = self.label
            if ignore_labels and 'label' in outv:
                outv.pop('label')
            return outv

REDUCTIONS = {
    'add': operator.add,
    'sum': operator.add,
    'multiply': operator.mul,
    'product': operator.mul,
    'prod': operator.mul,
    'subtract': operator.sub,
    'sub': operator.sub,
    'divide': operator.truediv,
    'div': operator.truediv,
    'truediv': operator.truediv,
    'max':max,
    'min':min,
    ## requires special treatment by caller
    'mean': operator.add,
    'average': operator.add,
    'avg': operator.add,
}


class Composition(ParameterGroup):

    def __init__(
        self,
        parameters:Union[Dict[str, Curve],'ParameterGroup'],
        weight:Optional[Union[Curve,Number]]=1,
        reduction:str=None,
        label:str=None,
        loop:bool=False,
        bounce:bool=False,
    ):
        self.loop, self.bounce = loop, bounce
        self.reduction = reduction
        super().__init__(parameters=parameters, weight=weight, label=label)
        # uh.... let's try this I guess?
        # uh... ok that seems to have fixed it. Interesting.
        if label is None:
            self.label = self.random_label()

    def __getitem__(self, k) -> Union[Number,dict]:

        k = self._adjust_k_for_looping(k)
        f = REDUCTIONS.get(self.reduction)

        vals = [curve[k] for curve in self.parameters.values()]
        outv = reduce(f, vals)
        if self.reduction in ('avg', 'average', 'mean'):
            outv = outv * (1/ len(vals))

        if self.weight != Curve({0:1}):
            outv = outv * self.weight[k]
        return outv

    def random_label(self, d=None) ->str:
        if d is None:
            d = self.parameters
        basename = ', '.join([str(keyname) for keyname in d.keys()])
        return f"{self.reduction}({basename})_{id_generator()}"

    def __sub__(self, other) -> 'Composition':

        if isinstance(other, ParameterGroup) and not isinstance(other, type(self)):
            return NotImplemented
        return super().__sub__(other)

    def __radd__(self, other) -> 'Composition':
        if isinstance(other, ParameterGroup) and not isinstance(other, type(self)):
            return NotImplemented
        return super().__radd__(other)

    def __add__(self, other) -> 'Composition':

        if isinstance(other, ParameterGroup) and not isinstance(other, type(self)):
            return NotImplemented

        if not isinstance(other, CurveBase):
            other = Curve(other)

        pg_copy = self.copy()
        if self.reduction in ('sum', 'add'):
            pg_copy.parameters[other.label] = other
            return pg_copy
        else:
            d = {pg_copy.label:pg_copy, other.label:other}
            return Composition(parameters=d, weight=pg_copy.weight, reduction='sum')

    def __mul__(self, other) -> 'ParameterGroup':
        if isinstance(other, ParameterGroup) and not isinstance(other, type(self)):
            return NotImplemented
        if not isinstance(other, CurveBase):
            other = Curve(other)

        pg_copy = self.copy()
        if self.reduction in ('multiply', 'mul', 'product', 'prod'):
            pg_copy.parameters[other.label] = other
            return pg_copy
        else:
            d = {pg_copy.label:pg_copy, other.label:other}
            return Composition(parameters=d, reduction='prod')

    def __truediv__(self, other) -> 'Composition':
        if isinstance(other, ParameterGroup) and not isinstance(other, type(self)):
            return NotImplemented
        if not isinstance(other, CurveBase):
            other = Curve(other)

        pg_copy = self.copy()
        d = {pg_copy.label:pg_copy, other.label:other}
        return Composition(parameters=d, reduction='truediv')
    
    def __rtruediv__(self, other) -> 'Composition':
        if isinstance(other, ParameterGroup) and not isinstance(other, type(self)):
            return NotImplemented
        if not isinstance(other, CurveBase):
            other = Curve(other)

        pg_copy = self.copy()
        d = {other.label:other, pg_copy.label:pg_copy} # reverse order of arguments
        return Composition(parameters=d, reduction='truediv')

    def plot(self, n:int=None, xs:list=None, eps:float=1e-9, *args, **kargs):

        is_compositional_pgroup = False
        for v in self.parameters.values():
            if isinstance(v, ParameterGroup) and not isinstance(v, Composition):
                is_compositional_pgroup = True
                break
        if not is_compositional_pgroup:
            Curve.plot(self, n=n, xs=xs, eps=eps, *args, **kargs)
        else: 
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise ImportError("Please install matplotlib to use Curve.plot()")

            if xs is None:
                if n is None:
                    n = self.duration + 1
                xs_base = list(range(int(n))) + list(self.keyframes)
                xs = set()
                for x in xs_base:
                    if (x>0) and (eps is not None) and (eps > 0):
                        xs.add(x-eps)
                    xs.add(x)
            xs = list(set(xs))
            xs.sort()
            
            ys = [self[x] for x in xs]

            ys_d =  {k: [dic[k] for dic in ys] for k in ys[0]}
            for label, values in ys_d.items():
                kargs['label'] = label
                plt.plot(xs, values, *args, **kargs)
                kfx = self.keyframes
                kfy = [self[x][label] for x in kfx]
                plt.scatter(kfx, kfy)
    def to_dict(self, simplify=False, for_yaml=False, ignore_labels=False):
        outv = super().to_dict(simplify=simplify, for_yaml=for_yaml, ignore_labels=ignore_labels)
        outv['reduction'] = self.reduction
        if ignore_labels and 'label' in outv:
            outv.pop('label')
        return outv



#endregion-----------------------依赖--------------------



#region-------------------------原代码--------------------


logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CATEGORY="Keyframe" + "/schedule"


class KfKeyframedCondition:

    CATEGORY=CATEGORY
    FUNCTION = 'main'
    RETURN_TYPES = ("KEYFRAMED_CONDITION",)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "time": ("FLOAT", {"default": 0, "step": 1}), 
                "interpolation_method": (list(EASINGS.keys()), {"default":"linear"}),
            },
        }
    
    def main(self, conditioning, time, interpolation_method):


        cond_tensor, cond_dict = conditioning[0] 
        cond_tensor = cond_tensor.clone()
        kf_cond_t = Keyframe(t=time, value=cond_tensor, interpolation_method=interpolation_method)

        cond_pooled = cond_dict.get("pooled_output")
        cond_dict = deepcopy(cond_dict)
        kf_cond_pooled = None
        if cond_pooled is not None:
            cond_pooled = cond_pooled.clone()
            kf_cond_pooled = Keyframe(t=time, value=cond_pooled, interpolation_method=interpolation_method)
            cond_dict["pooled_output"] = cond_pooled
        
        return ({"kf_cond_t":kf_cond_t, "kf_cond_pooled":kf_cond_pooled, "cond_dict":cond_dict},)


def set_keyframed_condition(keyframed_condition, schedule=None):
    keyframed_condition = deepcopy(keyframed_condition)
    cond_dict = keyframed_condition.pop("cond_dict")
    #cond_dict = deepcopy(cond_dict)

    if schedule is None:
        # get a new copy of the tensor
        kf_cond_t = keyframed_condition["kf_cond_t"]
        #kf_cond_t.value = kf_cond_t.value.clone() # should be redundant with the deepcopy
        curve_tokenized = Curve([kf_cond_t], label="kf_cond_t")
        curves = [curve_tokenized]
        if keyframed_condition["kf_cond_pooled"] is not None:
            kf_cond_pooled = keyframed_condition["kf_cond_pooled"]
            curve_pooled = Curve([kf_cond_pooled], label="kf_cond_pooled")
            curves.append(curve_pooled)
        schedule = (ParameterGroup(curves), cond_dict)
    else:
        schedule = deepcopy(schedule)
        schedule, old_cond_dict = schedule
        for k, v in keyframed_condition.items():
            if (v is not None):
                # for now, assume we already have a schedule for k.
                # Not sure how to handle new conditioning type appearing.
                schedule.parameters[k][v.t] = v
        old_cond_dict.update(cond_dict) # NB: mutating this is probably bad
        schedule = (schedule, old_cond_dict)
    return schedule





class KfSetKeyframe:
    CATEGORY=CATEGORY
    FUNCTION = 'main'
    RETURN_TYPES = ("SCHEDULE",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframed_condition": ("KEYFRAMED_CONDITION", {}),
            },
            "optional": {
                "schedule": ("SCHEDULE", {}), 
            }
        }
    def main(self, keyframed_condition, schedule=None):
        schedule = set_keyframed_condition(keyframed_condition, schedule)
        return (schedule,)


def evaluate_schedule_at_time(schedule, time):
    schedule = deepcopy(schedule)
    schedule, cond_dict = schedule
    #cond_dict = deepcopy(cond_dict)
    values = schedule[time]
    cond_t = values.get("kf_cond_t")
    cond_pooled = values.get("kf_cond_pooled")
    if cond_pooled is not None:
        #cond_dict = deepcopy(cond_dict)
        cond_dict["pooled_output"] = cond_pooled #.clone()
    #return [(cond_t.clone(), cond_dict)]
    return [(cond_t, cond_dict)]


class KfGetScheduleConditionAtTime:
    CATEGORY=CATEGORY
    FUNCTION = 'main'
    RETURN_TYPES = ("CONDITIONING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("SCHEDULE",{}),
                "time": ("FLOAT",{}),
            }
        }
    
    def main(self, schedule, time):
        lerped_cond = evaluate_schedule_at_time(schedule, time)
        return (lerped_cond,)


class KfGetScheduleConditionSlice:
    CATEGORY=CATEGORY
    FUNCTION = 'main'
    RETURN_TYPES = ("CONDITIONING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("SCHEDULE",{}),
                "start": ("FLOAT",{"default":0}),
                #"stop": ("FLOAT",{"default":0}),
                "step": ("FLOAT",{"default":1}),
                "n": ("INT", {"default":24}),
                #"endpoint": ("BOOL", {"default":True})
            }
        }
    
    #def main(self, schedule, start, stop, n, endpoint):
    def main(self, schedule, start, step, n):
        stop = start+n*step
        times = np.linspace(start=start, stop=stop, num=n, endpoint=True)
        conds = [evaluate_schedule_at_time(schedule, time)[0] for time in times]
        lerped_tokenized = [c[0] for c in conds]
        lerped_pooled = [c[1]["pooled_output"] for c in conds]
        lerped_tokenized_t = torch.cat(lerped_tokenized, dim=0)
        out_dict = deepcopy(conds[0][1])
        if isinstance(lerped_pooled[0], torch.Tensor) and isinstance(lerped_pooled[-1], torch.Tensor):
            out_dict['pooled_output'] =  torch.cat(lerped_pooled, dim=0)
        return [[(lerped_tokenized_t, out_dict)]] #


class AD_Evaluate_Condi:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
                "conditioning": ("CONDITIONING", {}),
                "interpolation_method": (list(EASINGS.keys()), {"default":"linear"}),
                "schedule": ("SCHEDULE",{}),
                "time": ("FLOAT", {"default": 0, "step": 1}), 
                "start": ("FLOAT",{"default":0}),
                "step": ("FLOAT",{"default":1}),
                "n": ("INT", {"default":60}),
                
                
            },

                "hidden": {
                
                    },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("condition", )
    FUNCTION = "main"
    CATEGORY = "Apt_Collect/AD"

    def main(self,conditioning, interpolation_method, schedule, time,start, step, n):

        cond_tensor, cond_dict = conditioning[0] 
        cond_tensor = cond_tensor.clone()
        kf_cond_t = Keyframe(t=time, value=cond_tensor, interpolation_method=interpolation_method)
        cond_pooled = cond_dict.get("pooled_output")
        cond_dict = deepcopy(cond_dict)
        kf_cond_pooled = None
        if cond_pooled is not None:
            cond_pooled = cond_pooled.clone()
            kf_cond_pooled = Keyframe(t=time, value=cond_pooled, interpolation_method=interpolation_method)
            cond_dict["pooled_output"] = cond_pooled
        
        keyframed_condition= {"kf_cond_t":kf_cond_t, "kf_cond_pooled":kf_cond_pooled, "cond_dict":cond_dict}    #算出keyframed_condition
        
        
        schedule = set_keyframed_condition(keyframed_condition, schedule)    #算出schedule
        
        stop = start+n*step        #估值
        times = np.linspace(start=start, stop=stop, num=n, endpoint=True)
        conds = [evaluate_schedule_at_time(schedule, time)[0] for time in times]
        lerped_tokenized = [c[0] for c in conds]
        lerped_pooled = [c[1]["pooled_output"] for c in conds]
        lerped_tokenized_t = torch.cat(lerped_tokenized, dim=0)
        out_dict = deepcopy(conds[0][1])
        if isinstance(lerped_pooled[0], torch.Tensor) and isinstance(lerped_pooled[-1], torch.Tensor):
            out_dict['pooled_output'] =  torch.cat(lerped_pooled, dim=0)
        return [[(lerped_tokenized_t, out_dict)]] 


#endregion--------------------收纳---------------------



#region-------------------------修改版本--------------



#region-------------------drawing schedule--------------------

from toolz.itertoolz import sliding_window

def schedule_to_weight_curves(schedule):
    schedule, _ = schedule
    schedule = schedule.parameters["kf_cond_t"]
    schedule = deepcopy(schedule)
    curves = []
    keyframes = list(schedule._data.values())
    
    if len(keyframes) == 1:
        keyframe = keyframes[0]
        curves = ParameterGroup({keyframe.label: keyframe.Curve(1)})
        return curves
    
    for (frame_in, frame_curr, frame_out) in sliding_window(3, keyframes):
        frame_in.value, frame_curr.value, frame_out.value = 0, 1, 0
        c = Curve({frame_in.t: frame_in, frame_curr.t: frame_curr, frame_out.t: frame_out}, 
                    label=frame_curr.label)
        c = deepcopy(c)
        curves.append(c)
    
    begin, end = keyframes[:2], keyframes[-2:]
    begin[0].value = 1
    begin[1].value = 0
    end[0].value = 0
    end[1].value = 1
    
    outv = [Curve(begin, label=begin[0].label)]
    if len(keyframes) == 2:
        return ParameterGroup({begin[0].label: outv[0]})
    
    outv += curves
    outv += [Curve(end, label=end[1].label)]
    return ParameterGroup({c.label: c for c in outv})


import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as TT

def plot_curve(curve, n, show_legend, is_pgroup=False):
    fig, ax = plt.subplots()
    eps: float = 1e-9

    m = 3
    if n < m:
        n = curve.duration + 1
        n = max(m, n)
    
    xs_base = list(range(int(n))) + list(curve.keyframes)
    logger.debug(f"xs_base:{xs_base}")
    xs = set()
    for x in xs_base:
        xs.add(x)
        xs.add(x - eps)

    width, height = 12, 8  # inches
    plt.figure(figsize=(width, height))        

    xs = [x for x in list(set(xs)) if (x >= 0)]
    xs.sort()

    def draw_curve(curve):
        ys = [curve[x] for x in xs]
        line = plt.plot(xs, ys, label=curve.label)
        kfx = curve.keyframes
        kfy = [curve[x] for x in kfx]
        plt.scatter(kfx, kfy, color=line[0].get_color())

    if is_pgroup:
        for c in curve.parameters.values():
            draw_curve(c)
    else:
        draw_curve(curve)
    
    if show_legend:
        plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()  # no idea if this makes a difference
    buf.seek(0)

    pil_image = Image.open(buf).convert('RGB')
    img_tensor = TT.ToTensor()(pil_image)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.permute([0, 2, 3, 1])
    return img_tensor


class AD_DrawSchedule:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "Apt_Preset/AD/😺backup"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("SCHEDULE", {"forceInput": True}),
                "n": ("INT", {"default": 64}),
                "show_legend": ("BOOLEAN", {"default": True}),
            }
        }

    def main(self, schedule, n, show_legend):
        curves = schedule_to_weight_curves(schedule)
        img_tensor = plot_curve(curves, n, show_legend, is_pgroup=True)
        return (img_tensor,)
#endregion----------------------drawing schedule--------------------



class AD_slice_Condi:

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive", )

    FUNCTION = "main"
    CATEGORY = "Apt_Preset/AD/😺backup"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframed_condition": ("KEYFRAMED_CONDITION", {}),
                "schedule": ("SCHEDULE",{}),
                "offset": ("INT",{"default":1}),
                #"step": ("FLOAT",{"default":1}),
                "total_flame": ("INT", {"default":30}),
            },
            "optional": {
                "schedule": ("SCHEDULE", {}), 
            }
        }
    

    def main(self, keyframed_condition, offset, schedule, total_flame):
        
        
        step=1
        n=total_flame
        
        schedule = set_keyframed_condition(keyframed_condition, schedule) #kfsetkeyframe
        stop = offset+n*step
        times = np.linspace(start=offset, stop=stop, num=n, endpoint=True)
        conds = [evaluate_schedule_at_time(schedule, time)[0] for time in times]
        lerped_tokenized = [c[0] for c in conds]
        lerped_pooled = [c[1]["pooled_output"] for c in conds]
        lerped_tokenized_t = torch.cat(lerped_tokenized, dim=0)
        out_dict = deepcopy(conds[0][1])
        if isinstance(lerped_pooled[0], torch.Tensor) and isinstance(lerped_pooled[-1], torch.Tensor):
            out_dict['pooled_output'] =  torch.cat(lerped_pooled, dim=0)
        
        positive= [[(lerped_tokenized_t, out_dict)]]
        
        
        return  positive


class AD_sch_prompt_adv(KfKeyframedCondition):

    CATEGORY = "Apt_Preset/AD/😺backup"
    FUNCTION = 'main'
    RETURN_TYPES = ("KEYFRAMED_CONDITION", "SCHEDULE",)
    RETURN_NAMES = ("keyframed_condition", "schedule", )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
                "context": ("RUN_CONTEXT",),

                "text1": ("STRING", {"multiline": True, "default": "hill"}),
                "keyframe1": ("FLOAT", {"default": 1, "step": 1}),
                "interpolation_method1": (list(EASINGS.keys()), {"default": "linear"}),

                "text2": ("STRING", {"multiline": True, "default": "sea"}),
                "keyframe2": ("FLOAT", {"default": 30, "step": 1}),
                "interpolation_method2": (list(EASINGS.keys()), {"default": "linear"}),

                "text3": ("STRING", {"multiline": True, "default": ""}),
                "keyframe3": ("FLOAT", {"default": 0, "step": 1}),
                "interpolation_method3": (list(EASINGS.keys()), {"default": "linear"}),

                "text4": ("STRING", {"multiline": True, "default": ""}),
                "keyframe4": ("FLOAT", {"default": 0, "step": 1}),
                "interpolation_method4": (list(EASINGS.keys()), {"default": "linear"}),

                "text5": ("STRING", {"multiline": True, "default": ""}),
                "keyframe5": ("FLOAT", {"default": 0, "step": 1}),
                "interpolation_method5": (list(EASINGS.keys()), {"default": "linear"}),

                "text6": ("STRING", {"multiline": True, "default": ""}),
                "keyframe6": ("FLOAT", {"default": 0, "step": 1}),
                "interpolation_method6": (list(EASINGS.keys()), {"default": "linear"}),
            },
            "optional": {
                "schedule": ("SCHEDULE", {}),
            }
        }

    def main(self, context, text1, keyframe1, interpolation_method1, text2, keyframe2, interpolation_method2,
                text3, keyframe3, interpolation_method3, text4, keyframe4, interpolation_method4,
                text5, keyframe5, interpolation_method5, text6, keyframe6, interpolation_method6, schedule=None):
        
        
        clip= context.get("clip")
        
        # 处理第一个关键帧
        tokens1 = clip.tokenize(text1)
        cond1, pooled1 = clip.encode_from_tokens(tokens1, return_pooled=True)
        conditioning = [[cond1, {"pooled_output": pooled1}]]
        keyframed_condition = super().main(conditioning, keyframe1, interpolation_method1)[0]  
        keyframed_condition["kf_cond_t"].label = text1
        schedule = set_keyframed_condition(keyframed_condition, schedule)

        # 处理第二个关键帧
        tokens2 = clip.tokenize(text2)
        cond2, pooled2 = clip.encode_from_tokens(tokens2, return_pooled=True)
        conditioning = [[cond2, {"pooled_output": pooled2}]]
        keyframed_condition = super().main(conditioning, keyframe2, interpolation_method2)[0]  
        keyframed_condition["kf_cond_t"].label = text2
        schedule = set_keyframed_condition(keyframed_condition, schedule)

        # 处理第三个关键帧
        tokens3 = clip.tokenize(text3)
        cond3, pooled3 = clip.encode_from_tokens(tokens3, return_pooled=True)
        conditioning = [[cond3, {"pooled_output": pooled3}]]
        keyframed_condition = super().main(conditioning, keyframe3, interpolation_method3)[0]  
        keyframed_condition["kf_cond_t"].label = text3
        schedule = set_keyframed_condition(keyframed_condition, schedule)

        # 处理第四个关键帧
        tokens4 = clip.tokenize(text4)
        cond4, pooled4 = clip.encode_from_tokens(tokens4, return_pooled=True)
        conditioning = [[cond4, {"pooled_output": pooled4}]]
        keyframed_condition = super().main(conditioning, keyframe4, interpolation_method4)[0]  
        keyframed_condition["kf_cond_t"].label = text4
        schedule = set_keyframed_condition(keyframed_condition, schedule)

        # 处理第五个关键帧
        tokens5 = clip.tokenize(text5)
        cond5, pooled5 = clip.encode_from_tokens(tokens5, return_pooled=True)
        conditioning = [[cond5, {"pooled_output": pooled5}]]
        keyframed_condition = super().main(conditioning, keyframe5, interpolation_method5)[0]  
        keyframed_condition["kf_cond_t"].label = text5
        schedule = set_keyframed_condition(keyframed_condition, schedule)

        # 处理第六个关键帧
        tokens6 = clip.tokenize(text6)
        cond6, pooled6 = clip.encode_from_tokens(tokens6, return_pooled=True)
        conditioning = [[cond6, {"pooled_output": pooled6}]]
        keyframed_condition = super().main(conditioning, keyframe6, interpolation_method6)[0]  
        keyframed_condition["kf_cond_t"].label = text6
        schedule = set_keyframed_condition(keyframed_condition, schedule)
        
        return (keyframed_condition, schedule, conditioning)


#endregion-------------------------修改版本-------------



