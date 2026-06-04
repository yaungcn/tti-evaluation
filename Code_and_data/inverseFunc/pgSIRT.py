# import matplotlib.pyplot as plt
import numpy as np
import os

# import pgcore
import pygimli as pg
import pygimli.meshtools as mt

# from pygimli.viewer.pv import drawSensors
from pygimli.physics.traveltime import TravelTimeManager


class Inverse_SIRT:
    def __init__(
        self,
        mesh_x,
        mesh_y,
        model_world,
        sensor_positions,
        refinement=1,
        dotsize=0.006,
        **kwargs,
    ):
        self.world = model_world
        self.sensor_positions = sensor_positions
        self.dotsize = dotsize
        self.refinement = refinement
        self.mesh = self.create_mesh(mesh_x, mesh_y)
        self.data = self.initialize_data_container()
        self.ttm = TravelTimeManager()
        self.inv_model = None
        self.inv_result = None
        # use **Kwargs to setup additional parameters
        self.kwargs = kwargs

    def create_mesh(self, x, y):
        return pg.meshtools.createGrid(x, y)

    def initialize_data_container(self):
        data = pg.physics.traveltime.DataContainerTT()
        for sensor in self.sensor_positions:
            data.createSensor(sensor)
        return data

    def setup_data(self, travel_times, source, ground_truth, error):
        self.data["s"] = source
        self.data["g"] = ground_truth
        self.data["t"] = travel_times
        self.data["err"] = error
        return self.data

    def D_data2txt(self, data, filename="inv_result.txt", dshape=[10, 10]):
        # f = open(filename, 'w')
        # size_data = data.shape()
        print("shape: ", dshape[0], dshape[1])
        n = dshape[0] * dshape[1]
        k = np.array([0.0] * n)
        j = 0
        for i in data:
            k[j] = i**2
            j = j + 1
        k = k.reshape(dshape[0], dshape[1])
        reversed_k = k  # [::-1]

        # print(k.shape)
        np.savetxt(filename, reversed_k)
        return reversed_k

    def export_to_tecplot(
        self,
        data,
        filename="inv_result.dat",
        variables="D",
        dx=10,
        dy=10,
        dotsize=0.006,
    ):
        k = data  # Reverse the order of rows for Tecplot compatibility
        with open(filename, "w") as f:
            f.write('TITLE = "Inversion Result"\n')
            f.write(f'VARIABLES = "X", "Y", "{variables}"\n')
            f.write(
                f"ZONE T='InvResult' ZONETYPE=FEQUADRILATERAL, \nN={(k.shape[1] + 1)*(k.shape[0] + 1)}, E={k.shape[0]*k.shape[1]}, \nDATAPACKING=BLOCK,\nVARLOCATION=([3]=CELLCENTERED)\n"
            )

            # Write X coordinates (vertices)
            for j in range(k.shape[0] + 1):
                for i in range(k.shape[1] + 1):
                    f.write(f"{i * dx} ")
                    f.write("\n")

            # Write Y coordinates (vertices)
            for j in range(k.shape[0] + 1):
                for i in range(k.shape[1] + 1):
                    f.write(f"{j * dy} ")
                    f.write("\n")

            # Write values (block centers)
            for j in range(k.shape[0]):
                for i in range(k.shape[1]):
                    f.write(f"{k[j, i]} ")
                    f.write("\n")

            # Write cell connectivity
            num_x = k.shape[1] + 1
            num_y = k.shape[0] + 1
            for j in range(k.shape[0]):
                for i in range(k.shape[1]):
                    # print(j,i)
                    f.write(
                        f"{(i+1)+j*num_x} {(i+2)+j*num_x} {(i+2)+(j+1)*num_x} {(i+1)+(j+1)*num_x}\n"
                    )

            # Write sensor positions
            for sensor in self.sensor_positions:
                f.write(
                    f"GEOMETRY T=circle C=black CS=GRID LT=0.1 FC=white X={sensor[0]:.5f} Y={sensor[1]:.5f} {self.dotsize}\n"
                )

    def run_inversion(
        self,
        mesh_x,
        mesh_y,
        datacontainer_dir,
        estDtxt_dir,
        estDdat_dir,
        verbose=True,
        secNodes=6,
        useGradient=False,
        vTop=1500,
        vBottom=3000,
        **kwargs,
    ):
        # Set up the TravelTimeManager
        self.ttm.inv.inv.setRecalcJacobian(True)
        self.data.save(datacontainer_dir)

        self.inv_model = self.ttm.invert(
            self.data,
            mesh=self.mesh,
            secNodes=secNodes,
            useGradient=useGradient,
            vTop=vTop,
            vBottom=vBottom,
            verbose=verbose,
            **kwargs,
        )
        if verbose:
            print("chi^2 = %.2f" % self.ttm.inv.chi2())

        # update inversion result
        self.inv_result = self.D_data2txt(
            self.inv_model, estDtxt_dir, [len(mesh_y) - 1, len(mesh_x) - 1]
        )


if __name__ == "__main__":
    pass
