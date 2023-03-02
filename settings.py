import numpy as np

forward_problem = {'angular_range': 180,
                   'angular_number': 120,
                   'noise_level': 0.01,
                   }

iCP_TV_algorithm = {'H': 10,
                   'K': np.concatenate((80 * np.ones((5, ), dtype=int), 30 * np.ones((5, ), dtype=int))),
                   'p': 1,
                   'lmbda': 1,
                   }

CP_TV_algorithm = {'H': 2,
                   'K': 350,
                   'p': 1,
                   'lmbda': 0.5,
                   }

iReNN_algorithm = {'H': 10,
                   'K': np.concatenate((80 * np.ones((5, ), dtype=int), 30 * np.ones((5, ), dtype=int))),
                   'iterative_schedule': np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0]),
                   'p': 1,
                   'lmbda': 1,
                   }