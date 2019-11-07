from functools import lru_cache
import numpy as np

@lru_cache(maxsize=8)
def filtergrid(size):
    """ Generates grid for constructing frequency domain filters
    
    Arguments:  rows, cols - Size of image/filter

    Returns:          x, y - Grids containing normalised frequency values
                             ranging from -0.5 to 0.5 in x and y directions
                             respectively. x and y are quadrant shifted.
    
    Inspired by filtergrid.m found at https://www.peterkovesi.com/matlabfns/
    """
    
    if type(size) is int:
        rows = cols = size
    
    else:
        rows = size[0]
        cols = size[1]
        
    range_1 = np.linspace(-(cols//2), np.floor((cols-1)/2), cols) / cols
    range_2 = np.linspace(-(rows//2), np.floor((rows-1)/2), rows) / rows
    
    x, y = np.meshgrid(range_1, range_2)
    
    #Quadrant shift so that filters are constructed with 0 frequency at the corners
    x = np.fft.ifftshift(x)
    y = np.fft.ifftshift(y)
    
    return x.T, y.T

@lru_cache(maxsize=8)
def radius_filtergrid(size):
    """
    
    """
    
    if type(size) is int:
        rows = cols = size
    
    else:
        rows = size[0]
        cols = size[1]
        
    x, y = filtergrid(size)
    radius = np.sqrt(x**2 + y**2)
    return radius

@lru_cache(maxsize=8)
def theta_filtergrid(size):
    """
    
    """
    
    if type(size) is int:
        rows = cols = size
    
    else:
        rows = size[0]
        cols = size[1]
        
    x, y = filtergrid(size)
    
    theta = np.arctan2(-y,x)            # Matrix values contain polar angle.
                                      # (note -ve y is used to give +ve
                                      # anti-clockwise angles)
    
    return theta

@lru_cache(maxsize=8)
def lowpassfilter(size, cutoff, order):
    """ Constructs a low-pass butterworth filter.
    
    param size: Two element vector specifying the size of filter to construct [rows cols].
    
    param cutoff: Cutoff frequency of the filter 0 - 0.5
    param order: Order of the filter, the higher n is the sharper the transition is. 
    (n must be an integer >= 1). Note that n is doubled so that it is always an even integer.

    f = 1/ (1 + (w/cutoff))^(2*order)

    The frequency origin of the returned filter is at the corners.
    
    Inspired by lowpassfilter.m found at https://www.peterkovesi.com/matlabfns/
    """
    
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError("The cutoff frequency must be between 0 and 0.5")
        
    if order != int(order) or order < 1:
        raise ValueError("order must be an iteger >= 1")
        
    if type(size) is int or type(size) is float:
        rows = cols = size
        
    else:
        rows = size[0]
        cols = size[1]
    
    #Construct spatial frequency values in terms of normalised radius from centre.
    radius = radius_filtergrid(size)
    
    lp_filter = 1.0 / (1.0 + (radius / cutoff)**(2*order))
    
    return lp_filter

@lru_cache(maxsize=8)
def _log_gabor_radial(size, wavelength, sigma):
    
    freq = 1.0/wavelength
    radius = radius_filtergrid(size)
    radius[0,0] = 1e-10
    radial_spread = np.exp((-(np.log(radius/freq))**2) / 
                    (2 * np.log(sigma)**2))    
    
    # Construct low-pass filter that is as large as possible, yet falls
    # away to zero at the boundaries. The radial part of the log Gabor 
    # filter is multiplied by the low-pass filter to ensure no extra frequencies 
    # at the 'corners' of the FFT are incorporated.
    lp = lowpassfilter(size,.45,15);  # Radius .45, 'sharpness' 15
    # Apply low-pass filter
    radial_spread = radial_spread * lp 
    # Set the value at the 0 frequency point of the filter 
    # back to zero (undo the radius fudge).
    radial_spread[0,0] = 0
    
    return radial_spread

@lru_cache(maxsize=8)
def _log_gabor_angular(size, angle, angular_frac):
    # For each point in the filter matrix calculate the angular distance from
    # the specified filter orientation.  To overcome the angular wrap-around
    # problem sine difference and cosine difference values are first computed
    # and then the atan2 function is used to determine angular distance.
    theta = theta_filtergrid(size)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    
    ds = sintheta * np.cos(angle) - costheta * np.sin(angle);    # Difference in sine.
    dc = costheta * np.cos(angle) + sintheta * np.sin(angle);    # Difference in cosine.
    dtheta = np.abs(np.arctan2(ds,dc));                          # Absolute angular distance.
    # Scale theta so that cosine spread function has the right wavelength and clamp to pi    
    dtheta = np.minimum(dtheta/angular_frac/2,np.pi)
    # The spread function is cos(dtheta) between -pi and pi.  We add 1,
    # and then divide by 2 so that the value ranges 0-1
    angular_spread = (np.cos(dtheta)+1)/2
    
    return angular_spread
    

def log_gabor_kernel(size, wavelength=3, sigma=0.55, angle=0.0, angular_frac=1/6, freq_domain=True):
    
    # Radial component which controls the frequency band that the filter responds to
    radial_spread = _log_gabor_radial(size, wavelength, sigma)    
    # The angular component, which controls the orientation that the filter responds to.
    angular_spread = _log_gabor_angular(size, angle, angular_frac)

    log_gabor_filter = radial_spread * angular_spread
    
    if not freq_domain:
        log_gabor_filter = np.fft.ifft2(log_gabor_filter)
    return log_gabor_filter

    
    
def log_gabor(image, wavelength=3, sigma=0.55, angle=0.0, angular_frac=1/6):
    image_fft = np.fft.fft2(image)
    
    kernel = log_gabor_kernel(image.shape, wavelength, sigma, angle, angular_frac)
    
    even_odd = np.fft.ifft2(image_fft*kernel)
    
    return even_odd

def mean_phase(image, min_wavelength = 3,sigma=0.55,n_scale=4, n_orient=6):
    phase_sum=np.zeros(image.shape)
    count = 0
    for scale in range(n_scale):
        wavelength = min_wavelength*2.1**scale
        for orient in range(n_orient):
            angle = orient*np.pi/n_orient
            eo = log_gabor(image, wavelength, sigma, angle=angle, angular_frac=1/n_orient)
            phase = np.arctan2(np.abs(eo.imag), eo.real)
            ampl = np.sqrt(eo.imag**2+eo.real**2)
            phase_sum += phase*ampl
            count += 1
            
    return phase_sum/count
