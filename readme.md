# Kohonen SOM API
This project uses a Self Organizing Map (SOM) to help visualize complex, high-dimensional data in a simple, 2D format. The goal is to group similar data together without requiring any external supervision.

This README highlights what was done in the old version and how the new changes improve the project. 

## Old Implementation:
* All the work was done in one long function called train. 
* The code used nested loops to update each node's weights. This approach worked for small datasets but could be slow on larger grids or data.

## New Changes and Improvements:
* All the SOM training logic is now encapsulated in a class called SOMAgent. This makes the code more organized

* Instead of using multiple nested loops for updating each node’s weight, the new code precomputes grid coordinates and uses NumPy’s vectorized operations. This was done by:
    Computing the Euclidean distances for all nodes at once using NumPy’s np.linalg.norm function on an array of precomputed grid coordinates. This is functionally the same metric (Euclidean distance), but it's done in a single, efficient vectorized operation instead of one node at a time.
Vectorized operations are like having a high-speed scanner that processes many items at once instead of checking them one by one. This makes the training process much faster, especially on larger grids or with more data.

* The new version uses Python's logging module to log progress (e.g., every 10 iterations).

### How to Use the New System?

`pip3 install -r requirements.txt`

To run the Application Locally and generate Batch Images & Start the Server:
Run the script:
                `python kohonen_app.py`
                
This will generate sample SOM images and save them under:
images/small (for a small 10x10 grid)
images/large (for a large 100x100 grid)


### For Developers
Modular and Maintainable Code:
The code is now easier to understand, test, and extend.

### Automated Tests and CI/CD:
Unit and integration tests ensure code quality and help catch issues early. The GitHub Actions workflow runs these tests automatically on each push or pull request.




