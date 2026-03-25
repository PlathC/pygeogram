# pygeogram - examples

This folder contains several examples for `geogram`'s python interface:

- `voronoi-2d` and `voronoi-3d` compute and plot Voronoi and Laguerre diagrams.
- `transport-2d` and `transport-3d` solves a semi discrete optimal transport problem in 2D and 3D.
- `fluid-2d` relies on a semi discrete optimal transport solver to implement a very simple fluid simulator.

## Prerequisites

To install the required dependencies, one can rely on the following command:

```sh
pip install -r requirements.txt
```

## Screenshots

- `voronoi-2d` and `voronoi-3d`

<center>
    <table>
        <tr>
            <td><img src="./img/voronoi-2d.png" alt="Voronoi 2D" style="height:200px; object-fit:contain;"/></td>
            <td><img src="./img/voronoi-3d.png" alt="Voronoi 3D" style="height:200px; object-fit:contain;"/></td>
        </tr>
    </table>
</center>

- `transport-2d` and `transport-3d`

<center>
    <table>
        <tr>
            <td><img src="./img/transport-2d.png" alt="Transport 2D" style="height:200px; object-fit:contain;"/></td>
            <td><img src="./img/transport-3d.png" alt="Transport 3D" style="height:200px; object-fit:contain;"/></td>
        </tr>
    </table>
</center>

- `fluid-2d`

<center>
    <table>
        <tr>
            <td><img src="./img/fluid-2d.png" alt="Fluid 2D" style="height:200px; object-fit:contain;"/></td>
        </tr>
    </table>
</center>