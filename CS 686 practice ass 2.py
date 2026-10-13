import numpy as np

# Variable indices (global)
#  Y  (0=young, 1=old)
#  D  (0=no disease, 1=disease)
#  M  (0=no med, 1=med)
#  S1 (0=no, 1=yes)
#  S2 (0=no, 1=yes)
#  S3 (0=no, 1=yes)

Y, D, M, S1, S2, S3 = 0, 1, 2, 3, 4, 5

# Variable Elimination Operations from the skeleton that was given 

def restrict(factor, variable, value):
    index = [slice(None)] * factor.ndim
    index[variable] = value
    restricted = factor[tuple(index)]
    restricted = np.expand_dims(restricted, axis=variable)
    return restricted

def sumout(factor, variable):

    return np.sum(factor, axis=variable, keepdims=True)

def multiply(factor1, factor2):
    return factor1 * factor2

def normalize(factor):
    total = np.sum(factor)
    if total == 0:
        return factor
    return factor / total

def inference(factor_list, query_variables, ordered_list_of_hidden_variables, evidence_list):

    
    for var, val in evidence_list:
        new_factors = []
        for f in factor_list:
            if f.shape[var] > 1:
                f = restrict(f, var, val)
            new_factors.append(f)
        factor_list = new_factors

    
    for Z in ordered_list_of_hidden_variables:
        if Z in query_variables:
            continue

        involved, remaining = [], []
        for f in factor_list:
            if f.shape[Z] > 1:
                involved.append(f)
            else:
                remaining.append(f)

        if involved:
            prod = involved[0]
            for f in involved[1:]:
                prod = multiply(prod, f)
            new_factor = sumout(prod, Z)
            remaining.append(new_factor)

        factor_list = remaining

    
    result = factor_list[0]
    for f in factor_list[1:]:
        result = multiply(result, f)


    return normalize(result)


#  Bayes net 

def build_medical_bn_factors():

    fY = np.array([0.8, 0.2]).reshape(2,1,1,1,1,1)
    fD = np.array([0.9, 0.1]).reshape(1,2,1,1,1,1)

    fM = np.zeros((2,1,2,1,1,1))
    fM[0,0,0,0,0,0] = 0.96
    fM[0,0,1,0,0,0] = 0.04
    fM[1,0,0,0,0,0] = 0.90
    fM[1,0,1,0,0,0] = 0.10

    fS1 = np.zeros((2,1,2,2,1,1))
    for y in [0,1]:
        for m in [0,1]:
            if m == 1:
                p = 0.20 if y == 1 else 0.05
            else:
                p = 0.02
            fS1[y,0,m,1,0,0] = p
            fS1[y,0,m,0,0,0] = 1 - p

    fS2 = np.zeros((1,2,2,1,2,1))
    for d in [0,1]:
        for m in [0,1]:
            if d == 1 and m == 0:
                p = 0.50
            elif d == 1 and m == 1:
                p = 0.10
            else:
                p = 0.01
            fS2[0,d,m,0,1,0] = p
            fS2[0,d,m,0,0,0] = 1 - p

    fS3 = np.zeros((1,2,1,1,1,2))
    for d in [0,1]:
        p = 0.20 if d == 1 else 0.05
        fS3[0,d,0,0,0,1] = p
        fS3[0,d,0,0,0,0] = 1 - p

    return [fY, fD, fM, fS1, fS2, fS3]


def run_query(query_vars, evidence, elim_order, factors=None):
    if factors is None:
        factors = build_medical_bn_factors()
    return inference(factors, query_vars, elim_order, evidence)

# Main code to check with written calculation

if __name__ == "__main__":

    # S1 was removed in the elimination order
    elim_order = [Y, M, S2, D, S3]

    # Prior 
    ans_prior = run_query([S1], [], elim_order)
    print("P(S1):", np.squeeze(ans_prior))

    # Posterior 
    ans_post = run_query([S1], [(S3,1),(S2,0)], elim_order)
    print("P(S1 | S3=1,S2=0):", np.squeeze(ans_post))

    # Observation
    ans_obs = run_query([S1], [(S3,1),(S2,0),(M,1)], elim_order)
    print("P(S1 | S3=1,S2=0,M=1):", np.squeeze(ans_obs))

    # Intervention
    factors_do = build_medical_bn_factors()
    fM_do = np.array([0.0, 1.0]).reshape(1,1,2,1,1,1)
    factors_do[2] = fM_do
    ans_do = inference(factors_do, [S1], elim_order, [(S3,1),(S2,0)])
    print("P(S1 | S3=1,S2=0,do(M=1)):", np.squeeze(ans_do))
