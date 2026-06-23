"""
Probability densities used to build the prior distributions.

These classes loosely mimic the scipy.stats API in their inputs and
functionality.

Original source of distribution setup: PBJam (https://github.com/PBjam-projects/PBjam)
"""

import jax.numpy as jnp
from functools import partial, cached_property
import jax
import numpy as np
import jax.scipy.special as jsp


class beta():
    def __init__(self, a=1, b=1, loc=0, scale=1):
        """ beta distribution class
        Create instances a probability density which follows the beta
        distribution.
        Parameters
        ----------
        a : float
            The first shape parameter of the beta distribution.
        b : float
            The second shape parameter of the beta distribution.
        loc : float
            The lower limit of the beta distribution. The probability at this
            limit and below is 0.
        scale : float
            The width of the beta distribution. Effectively sets the upper
            bound for the distribution, which is loc+scale.
        """

        # Turn init args into attributes
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.logfac = jsp.gammaln(self.a + self.b) - jsp.gammaln(self.a) - jsp.gammaln(self.b) - jnp.log(self.scale)

        self.fac = jnp.exp(self.logfac)

        self.am1 = self.a - 1

        self.bm1 = self.b - 1

    def rvs(self, size=None):
        """ Draw random variable from distribution

        Returns
        -------
        x : float
            Random variable drawn from the distribution
        """

        u = np.random.uniform(0, 1, size)
        
        x = self.ppf(u)
        
        return x
    
    @cached_property
    def mean(self):
        """ Distribution mean, computed lazily on first access. """
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1-1e-6), 1000)
        return jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)

    @cached_property
    def median(self):
        """ Distribution median, computed lazily on first access. """
        return self.ppf(0.5)

    @partial(jax.jit, static_argnums=(0,))
    def _transformx(self, x):
        """ Transform x
        Translates and scales the input x to the unit interval according to
        the loc and scale parameters.
        
        Parameters
        ----------
        x : float
            Input support for the probability density.
        
        Returns
        -------
        _x : float
            x translated and scaled to the range 0 to 1.
        """
        return (x - self.loc) / self.scale

    @partial(jax.jit, static_argnums=(0,))
    def _inverse_transform(self, x):
        """ Invert scaling on input

        Parameters
        ----------
        x : float
            Input

        Returns
        -------
        _x : float
            Scaled x. 
        """
        
        return x * self.scale + self.loc
 
    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x, norm=True):
        """ Return PDF

        Returns the beta distribution at x. The distribution is normalized to
        unit integral by default so that it may be used as a PDF.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized beta distribution. The default is
            True.
        
        Returns
        -------
        y : array
            The value of the beta distribution at x.
        """
 
        _x = self._transformx(x)
   
        T = jax.lax.lt(_x, 0.) | jax.lax.lt(1., _x)  
         
        y = jax.lax.cond(T, lambda : -jnp.inf, lambda : _x**self.am1 * (1 - _x)**self.bm1)
                
        if norm:
            return y * self.fac

        else:
            return y

        
    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x, norm=True):
        """ Return log-PDF
        Returns the log of the beta distribution at x. The distribution is
        normalized to unit integral (in linear units) by default so that it
        may be used as a PDF.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized beta distribution. The default is
            True.
        Returns
        -------
        y : array
            The value of the logarithm of the beta distribution at x.
        """

        x = jnp.array(x)

        _x = self._transformx(x)
            
        T = jax.lax.lt(_x, 0.) | jax.lax.lt(1., _x)  

        y = jax.lax.cond(T, lambda : -jnp.inf, lambda : self.am1 * jnp.log(_x) + self.bm1 * jnp.log(1-_x))
        
        if norm:
            return y + self.logfac
        else:
            return y

        
    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x):

        _x = jnp.asarray(self._transformx(x))

        y = jsp.betainc(self.a, self.b, _x)

        y = jnp.where(_x <= 0, 0.0, jnp.where(_x >= 1, 1.0, y))

        return y

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):

        _x = self.betaincinv(self.a, self.b, y)

        x = self._inverse_transform(_x)

        return x


    @partial(jax.jit, static_argnums=(0,))
    def update_x(self, x, a, b, p, a1, b1, afac):
        err = jsp.betainc(a, b, x) - p
        t = jnp.exp(a1 * jnp.log(x) + b1 * jnp.log(1.0 - x) + afac)
        u = err/t
        tmp = u * (a1 / x - b1 / (1.0 - x))
        t = u/(1.0 - 0.5 * jnp.clip(tmp, max=1.0))
        x -= t
        x = jnp.where(x <= 0., 0.5 * (x + t), x)
        x = jnp.where(x >= 1., 0.5 * (x + t + 1.), x)

        return x, t

    @partial(jax.jit, static_argnums=(0,))
    def func_1(self, a, b, p):
        pp = jnp.where(p < .5, p, 1. - p)
        t = jnp.sqrt(-2. * jnp.log(pp))
        x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
        x = jnp.where(p < .5, -x, x)
        al = (jnp.power(x, 2) - 3.0) / 6.0
        h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
        w = (x * jnp.sqrt(al + h) / h)-(1.0 / (2.0 * b - 1) - 1.0/(2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h))
        return a / (a + b * jnp.exp(2.0 * w))

    @partial(jax.jit, static_argnums=(0,))
    def func_2(self, a, b, p):
        lna = jnp.log(a / (a + b))
        lnb = jnp.log(b / (a + b))
        t = jnp.exp(a * lna) / a
        u = jnp.exp(b * lnb) / b
        w = t + u

        return jnp.where(p < t/w, jnp.power(a * w * p, 1.0 / a), 1. - jnp.power(b *w * (1.0 - p), 1.0/b))

    @partial(jax.jit, static_argnums=(0,))
    def compute_x(self, p, a, b):
        return jnp.where(jnp.logical_and(a >= 1.0, b >= 1.0), self.func_1(a, b, p), self.func_2(a, b, p))

    @partial(jax.jit, static_argnums=(0,))
    def betaincinv(self, a, b, p):
        a1 = a - 1.0
        b1 = b - 1.0

        ERROR = 1e-8

        p = jnp.clip(p, min=0., max=1.)

        x = jnp.where(jnp.logical_or(p <= 0.0, p >= 1.), p, self.compute_x(p, a, b))

        afac = - jsp.betaln(a, b)
        stop  = jnp.logical_or(x == 0.0, x == 1.0)
        for i in range(10):
            x_new, t = self.update_x(x, a, b, p, a1, b1, afac)
            x = jnp.where(stop, x, x_new)
            stop = jnp.where(jnp.logical_or(jnp.abs(t) < ERROR * x, stop), True, False)

        return x

class distribution():
    def __init__(self, ppf, pdf, logpdf, cdf):
        """ Generic distribution object

        Creates wrapper for a set of methods that return the pdf logpdf, ppf and 
        cdf of an arbitrary distribution. 
    
        Parameters
        ----------
        ppf : callable 
            Function that, given a value between 0 and 1, returns a sample drawn 
            from the pdf.
        pdf : callable
            Function that, given x returns the value of pdf(x)
        logpdf : callable
            Function that, given x return the value of log(pdf(x)). 
        cdf : callable
            Function that, given x returns the value of cdf(x).

        """

        self.pdf = pdf

        self.ppf = ppf

        self.logpdf = logpdf

        self.cdf = cdf

    def rv(self):
        """ Draw random variable from distribution

        Returns
        -------
        x : float
            Random variable drawn from the distribution
        """

        u = np.random.uniform(0, 1)
        
        x = self.ppf(u)
        
        return x
    
    @cached_property
    def mean(self):
        """ Distribution mean, computed lazily on first access. """
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1-1e-6), 1000)
        return jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)

    @cached_property
    def median(self):
        """ Distribution median, computed lazily on first access. """
        return self.ppf(0.5)

class uniform():
    def __init__(self, loc=0, scale=1):
        """ Uniform distribution

        Emulates the scipy.stats class, but is jaxed.

        Parameters
        ----------
        loc : float
            Left side of the uniform distribution
        scale : float
            Width of the uniform distribution, such that the right side is loc+scale
        
        Attributes
        ----------
        mu : float
            Mean (loc+scale/2) of the distribution.
        """

        # Turn init args into attributes
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
                
        self.a = self.loc

        self.b = self.loc + self.scale

    
    def rvs(self, size=None):
        """ Draw random variable from distribution

        Returns
        -------
        x : float
            Random variable drawn from the distribution
        """

        u = np.random.uniform(0, 1, size)
        
        x = self.ppf(u)
        
        return x
    
    @cached_property
    def mean(self):
        """ Distribution mean, computed lazily on first access. """
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1-1e-6), 1000)
        return jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)

    @cached_property
    def median(self):
        """ Distribution median, computed lazily on first access. """
        return self.ppf(0.5)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x):
        """ The probability density of the distribution

        Parameters
        ----------
        x : float
            Evaluate the pdf at x

        Returns
        -------
        y : float
            The probability at x
        """
            
        T = jax.lax.lt(x, self.a) | jax.lax.lt(self.b, x)  
             
        y = jax.lax.cond(T, lambda : 0., lambda : 1./self.scale)
            
        return y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x):
        """ The log-probability of the distribution

        Parameters
        ----------
        x : float
            Evaluate the pdf at x

        Returns
        -------
        y : float
            The probability at x
        """
 
        T = jax.lax.lt(x, self.a) | jax.lax.lt(self.b, x)  
            
        y = jax.lax.cond(T, lambda : -jnp.inf, lambda : -jnp.log(self.scale))
        
        return y

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x):
        """ The cumulative probability distribution function

        Parameters
        ----------
        x : float
            Evaluate the cdf at x

        Returns
        -------
        y : float
            The cumulative probability at x
        """

        y = (x - self.a) / (self.b - self.a)

        return jnp.clip(y, 0.0, 1.0)

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):
        """ The point percent (quantile) function. 

        Parameters
        ----------
        y : float
            Evaluate the ppf at y.

        Returns
        -------
        x : float
            The support of the pdf at pdf = y.
        """

        y = jnp.array(y)

        x = y * (self.b - self.a) + self.a

        return x

class normal():
    def __init__(self, loc=0, scale=1):
        """ normal distribution class

        Create instances a probability density which follows the normal
        distribution.

        Parameters
        ----------

        mu : float
            The mean of the normal distribution.
        sigma : float
            The standard deviation of the normal distribution.
        """
        # Turn init args into attributes
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.fac = -0.5 / self.scale**2      

        self.norm = 1 / (jnp.sqrt(2*jnp.pi) * self.scale)

        self.lognorm = jnp.log(self.norm)


    def rvs(self, size=None):
        """ Draw random variable(s) from distribution

        Returns
        -------
        x : float
            Random variable drawn from the distribution
        """

        u = np.random.uniform(0, 1, size)
        
        x = self.ppf(u)
        
        return x
    
    @cached_property
    def mean(self):
        """ Distribution mean, computed lazily on first access. """
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1-1e-6), 1000)
        return jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)

    @cached_property
    def median(self):
        """ Distribution median, computed lazily on first access. """
        return self.ppf(0.5)
    
    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x, norm=True):
        """ Return PDF

        Returns the normal distribution at x. The distribution is normalized to
        unit integral by default so that it may be used as a PDF.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized normal distribution. The default is
            True.

        Returns
        -------
        y : array
            The value of the normal distribution at x.

        """
        y = jnp.exp( self.fac * (x - self.loc)**2)

        Y = jax.lax.cond(norm,
                         lambda y: y * self.norm,
                         lambda y: y ,
                         y)

        return Y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x, norm=True):
        """ Return log-PDF

        Returns the log of the normal distribution at x. The distribution is
        normalized to unit integral (in linear units) by default so that it
        may be used as a PDF.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized normal distribution. The default is
            True.

        Returns
        -------
        y : array
            The value of the logarithm of the normal distribution at x.
        """

        y = self.fac * (x - self.loc)**2

        Y = jax.lax.cond(norm,
                         lambda y: y + self.lognorm,
                         lambda y: y,
                         y)

        return Y

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x):

        y = 0.5 * (1 + jsp.erf((x-self.loc)/(jnp.sqrt(2)*self.scale)))

        return y

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):

        x = self.loc + self.scale*jnp.sqrt(2)*jsp.erfinv(2*y-1)

        return x

class truncsine():
    def __init__(self,):
        """ Sine truncated between 0 and pi/2
        """
        
    def rv(self):
        """ Draw random variable from distribution

        Returns
        -------
        x : float
            Random variable drawn from the distribution
        """

        u = np.random.uniform(0, 1)
        
        x = self.ppf(u)
        
        return x

    @cached_property
    def mean(self):
        """ Distribution mean, computed lazily on first access. """
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1-1e-6), 1000)
        return jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)

    @cached_property
    def median(self):
        """ Distribution median, computed lazily on first access. """
        return self.ppf(0.5)
 
    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x):
        """ The probability density of the distribution

        Parameters
        ----------
        x : float
            Evaluate the pdf at x

        Returns
        -------
        y : float
            The probability at x
        """

        T = jax.lax.lt(x, 0.) | jax.lax.lt(jnp.pi/2, x)  
             
        y = jax.lax.cond(T, lambda : 0., lambda : jnp.sin(x))
            
        return y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x):
        """ The log-probability of the distribution

        Parameters
        ----------
        x : float
            Evaluate the pdf at x

        Returns
        -------
        y : float
            The probability at x
        """

        T = jax.lax.lt(x, 0.) | jax.lax.lt(jnp.pi/2., x)  
             
        y = jax.lax.cond(T, lambda : -jnp.inf, lambda : jnp.log(jnp.sin(x)))

        return y

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x):
        """ The cumulative probability distribution function

        Parameters
        ----------
        x : float
            Evaluate the cdf at x

        Returns
        -------
        y : float
            The cumulative probability at x
        """

        y = 1 + jnp.cos(x - jnp.pi)

        return jnp.where(x < 0, 0.0, jnp.where(x > jnp.pi / 2, 1.0, y))

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):
        """ The point percent (quantile) function. 

        Parameters
        ----------
        y : float
            Evaluate the ppf at y.

        Returns
        -------
        x : float
            The support of the pdf at pdf = y.
        """

        x = jnp.arccos(1-y)

        return x
    
class randint():
    def __init__(self, low, high):
        self.low = low
        
        self.high = high
        
        self.diff = self.high - self.low

        self.ints = jnp.arange(low, high, 1)

    def rv(self):
        """ Draw random variable from distribution

        Returns
        -------
        x : float
            Random variable drawn from the distribution
        """

        u = np.random.uniform(0, 1)
        
        x = self.ppf(u)
        
        return x

    @cached_property
    def mean(self):
        """ Distribution mean, computed lazily on first access. """
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1-1e-6), 1000)
        return jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)

    @cached_property
    def median(self):
        """ Distribution median, computed lazily on first access. """
        return self.ppf(0.5)

    @partial(jax.jit, static_argnums=(0,))    
    def pdf(self, x):
        """ Probability mass at x (1/diff on the integer support, else 0). """
        T = jnp.sum(self.ints == x).astype(bool)
        
        y = jax.lax.cond(T, lambda : 1/self.diff, lambda : 0.)

        return y
    
    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x):
        T = jnp.sum(self.ints == x).astype(bool)

        y = jax.lax.cond(T, lambda : -jnp.log(self.diff), lambda : -jnp.inf)

        return y

    @partial(jax.jit, static_argnums=(0,))    
    def cdf(self, x):
        
        k = jnp.floor(x)
        
        return (k - self.low + 1.) / self.diff
        
    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, q):
        vals = jnp.ceil(q * self.diff + self.low) - 1
        
        vals1 = (vals - 1).clip(self.low, self.high)
        
        temp = self.cdf(vals1)
         
        return jnp.floor(jnp.where(temp >= q, vals1, vals))
