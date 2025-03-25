#include "readOFF.h"
#include "mesh.h"
#include "serialization.h"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <filesystem>
#include <string>

using namespace Eigen;
using namespace std;

double tolerance = 1e-3;

namespace fs = std::filesystem;

double max_sparse(const Eigen::SparseMatrix<double>& mat, int &rowIndex, int &colIndex){
    double maxValue = -1;
    for (int k=0; k<mat.outerSize(); ++k)
        for (SparseMatrix<double>::InnerIterator it(mat,k); it; ++it)
        {
            if (std::isnan(it.value())){
                rowIndex  = it.row();
                colIndex = it.col();
                return std::numeric_limits<double>::quiet_NaN();
            }

            if (abs(it.value())>maxValue){
                maxValue = it.value();
                rowIndex  = it.row();
                colIndex = it.col();
            }
        }
    return maxValue;
}

// Helper function to compare two matrices and print differences
void compareSparseMatrices(const string& name, const SparseMatrix<double>& computed,
                          const SparseMatrix<double>& groundTruth, double tolerance) {
    int rowIndex, colIndex;
    double maxValue = max_sparse(groundTruth-computed, rowIndex, colIndex);

    cout << "Checking " << name << " matrix..." << endl;

    if ((maxValue <= tolerance) && (!std::isnan(maxValue))) {
        cout << name << " is good!" << endl;
    } else {
        cout << name << "(" << rowIndex << "," << colIndex << ")="
             << computed.coeff(rowIndex, colIndex) << ", Ground-truth "
             << name << "(" << rowIndex << "," << colIndex << ")="
             << groundTruth.coeff(rowIndex, colIndex) << endl;

        // Additional debug info for this element
        cout << "Difference at (" << rowIndex << "," << colIndex << "): "
             << groundTruth.coeff(rowIndex, colIndex) - computed.coeff(rowIndex, colIndex) << endl;
        cout << "Ratio: " << groundTruth.coeff(rowIndex, colIndex) / computed.coeff(rowIndex, colIndex) << endl;

        // Check if the matrices have the same sparsity pattern
        bool computedHasValue = computed.coeff(rowIndex, colIndex) != 0;
        bool groundTruthHasValue = groundTruth.coeff(rowIndex, colIndex) != 0;

        if (computedHasValue != groundTruthHasValue) {
            cout << "Sparsity pattern mismatch at (" << rowIndex << "," << colIndex << ")" << endl;
        }
    }
}

int main()
{
    // Default values
    double youngModulus = 10000;
    double PoissonRatio = 0.3;
    double density = 2.5;
    double alpha = 0.1, beta = 0.1, timeStep = 0.02;

    // Set the specific mesh file to debug here (leave empty to test all meshes)
    string specificMeshFile = "box_tri.mesh";
    //string specificMeshFile = "";  // Uncomment to test all meshes

    if (!specificMeshFile.empty()) {
        cout << "Testing specific mesh file: " << specificMeshFile << endl;
    }

    std::string folderPath(DATA_PATH);

    // Process either the specific file or all files
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mesh") {
            string filename = entry.path().filename().string();

            // Skip if we're looking for a specific file and this isn't it
            if (!specificMeshFile.empty() && filename != specificMeshFile) {
                continue;
            }

            cout << "\n==== Working on file \"" << filename << "\" ====" << endl;
            std::string dataName = entry.path().string();
            dataName.erase(dataName.size() - 5, 5);
            std::ifstream ifs(dataName+"-section1.data", std::ofstream::binary);

            if (!ifs.is_open()) {
                cout << "Error: Could not open ground truth data file: " << dataName << "-section1.data" << endl;
                continue;
            }

            MatrixXd objV;
            MatrixXi objF, objT;
            readMESH(entry.path().string(), objV, objF, objT);
            MatrixXi tempF(objF.rows(), 3);
            tempF << objF.col(2), objF.col(1), objF.col(0);
            objF = tempF;

            VectorXd Vxyz(3*objV.rows());
            for (int i=0; i<objV.rows(); i++)
                Vxyz.segment(3*i, 3) = objV.row(i).transpose();

            Mesh m(Vxyz, objF, objT, 0, youngModulus, PoissonRatio, density, false, RowVector3d::Zero(), {1.0,0.0,0.0,0.0});

            auto start = std::chrono::high_resolution_clock::now();
            m.create_global_matrices(timeStep, alpha, beta);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            cout << "create_global_matrices() took " << (double)(duration.count())/1000.0 << " seconds to execute." << endl;

            // Load ground truth matrices
            SparseMatrix<double> MGT, KGT, DGT;
            deserializeSparseMatrix(MGT, ifs);
            deserializeSparseMatrix(KGT, ifs);
            deserializeSparseMatrix(DGT, ifs);

            // Print matrix dimensions for debugging
            cout << "Matrix dimensions:" << endl;
            cout << "M: " << m.M.rows() << " x " << m.M.cols() << " (nnz: " << m.M.nonZeros() << ")" << endl;
            cout << "K: " << m.K.rows() << " x " << m.K.cols() << " (nnz: " << m.K.nonZeros() << ")" << endl;
            cout << "D: " << m.D.rows() << " x " << m.D.cols() << " (nnz: " << m.D.nonZeros() << ")" << endl;
            cout << "MGT: " << MGT.rows() << " x " << MGT.cols() << " (nnz: " << MGT.nonZeros() << ")" << endl;
            cout << "KGT: " << KGT.rows() << " x " << KGT.cols() << " (nnz: " << KGT.nonZeros() << ")" << endl;
            cout << "DGT: " << DGT.rows() << " x " << DGT.cols() << " (nnz: " << DGT.nonZeros() << ")" << endl;

            // Compare matrices
            compareSparseMatrices("M", m.M, MGT, tolerance);
            compareSparseMatrices("K", m.K, KGT, tolerance);
            compareSparseMatrices("D", m.D, DGT, tolerance);

            // If we're testing a specific file, add more detailed debugging
            if (!specificMeshFile.empty()) {
                // Debug information for the first few tetrahedra
                cout << "\nTetrahedron Debug Info (first 3 tets):" << endl;
                int maxTetsToDebug = min(3, (int)objT.rows());

                for (int t = 0; t < maxTetsToDebug; t++) {
                    cout << "Tet " << t << " vertices: "
                         << objT(t, 0) << ", " << objT(t, 1) << ", "
                         << objT(t, 2) << ", " << objT(t, 3) << endl;

                    Vector3d v0 = Vxyz.segment<3>(3 * objT(t, 0));
                    Vector3d v1 = Vxyz.segment<3>(3 * objT(t, 1));
                    Vector3d v2 = Vxyz.segment<3>(3 * objT(t, 2));
                    Vector3d v3 = Vxyz.segment<3>(3 * objT(t, 3));

                    Matrix3d Dm;
                    Dm.col(0) = v1 - v0;
                    Dm.col(1) = v2 - v0;
                    Dm.col(2) = v3 - v0;

                    Vector3d e01 = v1 - v0;
                    Vector3d e02 = v2 - v0;
                    Vector3d e03 = v3 - v0;

                    double volMethod1 = std::abs(Dm.determinant()) / 6.0;
                    double volMethod2 = std::abs(e01.dot(e02.cross(e03))) / 6.0;

                    cout << "  Volume (det method): " << volMethod1 << endl;
                    cout << "  Volume (cross method): " << volMethod2 << endl;
                    cout << "  Det(Dm): " << Dm.determinant() << endl;
                    cout << "  Condition number: " << Dm.norm() * Dm.inverse().norm() << endl;
                }
            }

            // If testing a specific file, only process that one
            if (!specificMeshFile.empty()) {
                break;
            }
        }
    }

    return 0;
}