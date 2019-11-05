
import neve
import numpy as np
import pandas as pd
import plotly.express as pe
import tqdm

def main():
    net, inp = neve.io.read_tf_file("mnist_relu_5_100.tf")

    data = np.load("mnist.npz")
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]

    result = net.forward({inp: x_test.reshape(-1, 784)})

    b = x_test[1].reshape(784)

    es = np.linspace(0, 25, 50)
    results = []
    for e in tqdm.tqdm(es):
        lo = np.maximum(b - e, 0)
        up = np.minimum(b + e, 255)

        #result = net.forward({inp: [b, lo, up]})
        state = neve.VerificationState({inp: [lo, up]})
        r_lo, r_up = net.compute_bounds(state)

        #for i in [2, 3, 4, 8]:
        #    if i == 2:
        #        results.append(("lo_{}".format(i), e, r_lo[i], "orig"))
        #    results.append(("up_{}".format(i), e, r_up[i], "orig"))


        """
        r_los = []
        r_ups = []
        for i in range(500):
            state = neve.VerificationState({inp: [lo, up]})
            state.mode = "new-rnd"
            r_lo, r_up = net.compute_bounds(state)
            r_los.append(r_lo)
            r_ups.append(r_up)

        r_lo = np.max(r_los, axis=1)
        r_up = np.min(r_ups, axis=1)
        """

        """
        for i in range(10): # [2, 3, 4, 8]:
            if i == 2:
                results.append(("lo_{}".format(i), e, r_lo[i], "rnd1"))
            results.append(("up_{}".format(i), e, r_up[i], "rnd1"))
        """

        for mode in ["new"]:
            state = neve.VerificationState({inp: [lo, up]})
            state.mode = mode
            r_lo, r_up = net.compute_bounds(state)

            for i in range(10): # [2, 3, 4, 8]:
                if i == 2:
                    results.append(("lo_{}".format(i), e, r_lo[i], mode))
                results.append(("up_{}".format(i), e, r_up[i], mode))

    df = pd.DataFrame(results, columns=["name", "e", "value", "mode"])

    fig = pe.line(df, x="e", y="value", color="name", line_dash="mode")
    fig.show("firefox")




    #print(result[1])
    #print(y_test[1])




if __name__ == "__main__":
    main()
