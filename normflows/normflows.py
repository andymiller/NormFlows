"""
Pure numpy/autograd implementation of normalizing flows
"""
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from autograd import grad, elementwise_grad
from autograd.optimizers import adam, sgd
from misc import WeightsParser, mog_logprob, kl_mvn_diag


##############################################
# define planar and real_nvp flow-makers     #
##############################################

def make_planar_transform():
    """normalized flow transformation: planar flow
        w \in R^D
        u \in R^D
        b \in R

        For invertibility, it is sufficient for dot(w, u) > -1
        to enforce this, we actually use a vector

            hat u(w, u) = u + [m(dot(w,u)) - dot(w,u)] * w / ||w||^2

        See Variational Inference with Normalizing Flows Appendix
        (http://jmlr.org/proceedings/papers/v37/rezende15-supp.pdf)
    """
    def uhat(u, w):
        wdir = w / np.dot(w,w)
        mwu  = -1 + np.log(1 + np.exp(np.dot(w, u)))
        return u + (mwu - np.dot(w,u)) * wdir

    def flow(z, w, b, u):
        hterm = np.tanh( np.dot(z, w) + b )
        return z + uhat(u, w) * hterm[:, None]

    def flow_det(z, w, b, u):
        x = np.dot(z, w) + b
        psi = elementwise_grad(np.tanh)(np.dot(z, w) + b)[:,None] * w
        return np.abs(1 + np.dot(psi, uhat(u, w)))

    return flow, flow_det


def make_real_nvp(d, D):
    """
    This defines a transformaiton using the Real NVP class of functions. Each
    transformation from D => D is defined conditionally; some set of
    d variables are held constant, and the other set are transformed
    conditioned on the first set (and a self-linear term)
    """

    # first define the conditioning set and non-conditioning set
    mask = np.zeros(D)
    if np.isscalar(d):
        mask[:d] = 1
    else:
        mask[d] = 1

    # make parameter vector
    params = np.concatenate([np.random.randn(D*D + D),
                             np.random.randn(D*D + D)])

    def unpack(params):
        W, b = params[:D*D], params[D*D:]
        return np.reshape(W, (D,D)), b

    def lfun(z, lparams):
        """ function from d => D-d, parameterized by lparams.  implemented
        as a full function, and we assume the input and output are masked """
        W, b = unpack(lparams)
        return np.tanh(np.dot(z, W) + b)

    def mfun(z, mparams):
        """ function from d => D-d, parameterized by lparams.  implemented
        as a full function, and we assume the input and output are masked """
        W, b = unpack(mparams)
        return np.tanh(np.dot(z, W) + b)

    def flow(z, params):
        lparams, mparams = np.split(params, 2)
        return mask*z + (1-mask)*(z*np.exp(lfun(mask*z, lparams)) + 
                                  mfun(mask*z, mparams))

    def flow_inv(zprime, params):
        lparams, mparams = np.split(params, 2)
        first_part = mask * zprime
        sec_part  = (1-mask) * (zprime - mfun(mask*zprime, mparams)) * \
                               np.exp(-lfun(mask*zprime,lparams))
        return first_part + sec_part

    def flow_det(z, params):
        """ log determinant of this flow """
        lparams, mparams = np.split(params, 2)
        diag = (1-mask)*lfun(mask*z,lparams)
        if len(z.shape) > 1:
            return np.sum(diag, axis=1)
        else:
            return np.sum(diag)

    return flow, flow_det, flow_inv, params


########################################
# variational objective function maker #
########################################


def make_variational_objective_funs(logprob, D, beta_schedule,
                                    num_layers = 2, debug_print=False):
    """ Normalized flow variational inference

      logprob = log p(x | z) p(z)

    """
    # add z ~ f(eps; theta) neural net parameters
    parser = WeightsParser()
    layers = []
    for l in range(num_layers):
        parser.add_shape("u_%d"%l, (D, ))
        parser.add_shape("w_%d"%l, (D, ))
        parser.add_shape("b_%d"%l, (1, ))
        layers.append(("u_%d"%l, "w_%d"%l, "b_%d"%l))

    # make planar transformation functions
    flow, flow_det = make_planar_transform()

    def feed_forward(z, params, layers):
        """ push z=z0 through the layers defined above """
        curr_units = z
        ldet_sum   = np.zeros(z.shape[0])
        for l, ln in enumerate(layers):
            ul = parser.get(params, ln[0])
            wl = parser.get(params, ln[1])
            bl = parser.get(params, ln[2])
            curr_units = flow(curr_units, wl, bl, ul)

            ldet_sum = ldet_sum + np.log(flow_det(curr_units, wl, bl, ul))

        return curr_units, ldet_sum

    def draw_variational_samples(params, num_samps, eps=None):
        # two step sampler
        if eps is None:
            eps = npr.randn(num_samps, D)
        #lleps = mvn.logpdf(eps, mean=np.zeros(D), cov=np.eye(D))
        zs, ldet_sums = feed_forward(eps, params, layers)
        return zs, ldet_sums

    def qlogprob(z0, params):
        zs, ldet_sums = feed_forward(eps, params, layers)

    def lnq_grid(params):
        assert D == 2
        xg, yg = np.linspace(-4, 4, 50), np.linspace(-4, 4, 50)
        xx, yy = np.meshgrid(xg, yg)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        zs, ldets = feed_forward(pts, params, layers)
        lls = mvn.logpdf(pts, mean=np.zeros(D), cov=np.eye(D)) - ldets
        return zs[:,0].reshape(xx.shape), zs[:,1].reshape(yy.shape), lls.reshape(xx.shape)

    def variational_objective(params, t, num_samples, beta=1.):
        """Provides a stochastic estimate of the variational lower bound."""

        # 1. draw samples from the variational posterior, eps ~ N(0,I)
        zs, ldet_sums = draw_variational_samples(params, num_samples)

        # 1.5 negative entropy of z0
        # not needed for optimization

        # 2. compute expected value of the sum of jacobian terms
        E_ldet_sum = np.mean(ldet_sums)

        # 3. compute data term
        lls = logprob(zs, t)
        E_logprob  = np.mean(lls)

        if debug_print:
            print "entropy term: ", E_ldet_sum
            print "data term   : ", E_logprob, " (+/- ", np.std(lls), ")", " min = ", np.min(lls)

        # return lower bound
        beta = 1. if t >= len(beta_schedule) else beta_schedule[t]
        lower_bound = beta * E_logprob + E_ldet_sum
        return -lower_bound

    gradient = grad(variational_objective)
    return variational_objective, gradient, parser.num_weights, \
           draw_variational_samples, lnq_grid


if __name__ == '__main__':
    npr.seed(42)

    #################################
    # set up a target density       #
    #################################
    D = 2
    K = 2
    icovs = np.zeros((K, D, D))
    dets, pis = np.zeros(K), np.zeros(K)
    means = np.array([[-1.5, 1.3], [1., -1.3]])
    for k in range(len(means)):
        icovs[k, :, :] = 1/.5*np.eye(D)
        dets[k]  = 1.
        pis[k] = .5

    # log unnormalized distribution to learn
    lnpdf = lambda z, t: mog_logprob(z, means, icovs, dets, pis)

    ########################################
    # Build variational objective/gradient #
    ########################################
    objective, gradient, num_variational_params, sample_z, lnq_grid = \
        make_variational_objective_funs(logprob = lnpdf,
                                        beta_schedule = np.linspace(1., 1., 10000),
                                        D=D, num_layers=8)

    #######################################################
    # set up plots + callback for monitoring optimization #
    #######################################################
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    from plots import plot_isocontours
    def make_plot_fun():
        num_plot_samps = 1000
        eps = npr.randn(num_plot_samps, D)
        def plot_q_dist(ax, params):
            zsamps, _ = sample_z(params, num_plot_samps, eps)
            print(" .... ", np.mean(zsamps, axis=0))
            xx, yy, ll = lnq_grid(params)
            ax.contour(xx, yy, np.exp(ll))
            #plt.scatter(zsamps[::2,0], zsamps[::2,1], c='grey', s=5, alpha=.5)
        return plot_q_dist

    plot_q_dist = make_plot_fun()

    # setup figure
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    def callback(params, t, g):
        print("Iteration {} lower bound {}".format(t, -objective(params, t, 1000)))
        if t % 10 == 0:
            plt.cla()
            # plot target
            target_distribution = lambda x : np.exp(lnpdf(x, t))
            plot_isocontours(ax, target_distribution, fill=True)
            # plot approximate distribution
            plot_q_dist(ax, params)
            ax.set_xlim((-3, 3))
            ax.set_ylim((-4, 4))
            plt.draw()
            plt.pause(1.0/30.0)

    #####################
    # Run optimization  #
    #####################
    print("Optimizing variational parameters...")
    th = .5*npr.randn(num_variational_params) - 3.

    num_objective_samps = 10
    def grad_wrap(th, t):
        return gradient(th, t, num_objective_samps)

    variational_params = adam(grad_wrap, th, step_size=.02, num_iters=10000,
                              callback=callback)

