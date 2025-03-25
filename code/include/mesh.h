#ifndef MESH_HEADER_FILE
#define MESH_HEADER_FILE

#include <vector>
#include <fstream>
#include "readMESH.h"
#include "auxfunctions.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;
using namespace std;


//the class the contains each individual rigid objects and their functionality
class Mesh{
public:
    
    //position
    VectorXd origPositions;     //3|V|x1 original vertex positions in xyzxyz format - never change this!
    VectorXd currPositions;     //3|V|x1 current vertex positions in xyzxyz format
    
    //kinematics
    bool isFixed;               //is the object immobile (infinite mass)
    VectorXd currVelocities;    //3|V|x1 velocities per coordinate in xyzxyz format.
    
    double totalInvMass;
    
    MatrixXi T;                 //|T|x4 tetrahdra
    MatrixXi F;                 //|F|x3 boundary faces
    VectorXd invMasses;         //|V|x1 inverse masses of vertices, computed in the beginning as 1.0/(density * vertex voronoi area)
    VectorXd voronoiVolumes;    //|V|x1 the voronoi volume of vertices
    VectorXd tetVolumes;        //|T|x1 tetrahedra volumes
    int globalOffset;           //the global index offset of the of opositions/velocities/impulses from the beginning of the global coordinates array in the containing scene class
    
    VectorXi boundTets;  //just the boundary tets, for collision
    
    double youngModulus, poissonRatio, density, alpha, beta;
    
    SparseMatrix<double> K, M, D;   //The soft-body matrices
    
    //SimplicialLLT<SparseMatrix<double>>* ASolver;   //the solver for the left-hand side matrix constructed for FEM
    
    ~Mesh(){/*if (ASolver!=NULL) delete ASolver;*/}
    
    
    
    bool isNeighborTets(const RowVector4i& tet1, const RowVector4i& tet2){
        for (int i=0;i<4;i++)
            for (int j=0;j<4;j++)
                if (tet1(i)==tet2(j)) //shared vertex
                    return true;
        
        return false;
    }
    

    //Computing the K, M, D matrices per mesh.
    void create_global_matrices(const double timeStep, const double _alpha, const double _beta)
    {

    // Store alpha and beta for damping matrix calculation
    alpha = _alpha;
    beta = _beta;

    // Calculate Lamé parameters
    double mu = youngModulus / (2 * (1 + poissonRatio));
    double lambda = (poissonRatio * youngModulus) / ((1 + poissonRatio) * (1 - 2 * poissonRatio));

    // Get dimensions of the system
    int dim = currVelocities.size();
    int numVertices = dim / 3;

    // Initialize triplets for all sparse matrices
    std::vector<Triplet<double>> KTriplets;
    std::vector<Triplet<double>> MTriplets;

    // Estimate number of non-zeros for stiffness matrix
    int estimatedNonZeros = T.rows() * 16 * 9;
    KTriplets.reserve(estimatedNonZeros);

    // For mass matrix (diagonal), we know exactly how many non-zeros: 3 per vertex
    MTriplets.reserve(dim);

    // -------------------- MASS MATRIX --------------------
    // Create lumped mass matrix (diagonal)
    for (int i = 0; i < numVertices; i++) {
        // Use inverse masses calculated in initializeVolumesAndMasses()
        double vertexMass = 1.0 / invMasses(i);

        // Each vertex has mass in x, y, and z components
        MTriplets.push_back(Triplet<double>(3*i, 3*i, vertexMass));
        MTriplets.push_back(Triplet<double>(3*i+1, 3*i+1, vertexMass));
        MTriplets.push_back(Triplet<double>(3*i+2, 3*i+2, vertexMass));
    }

    // -------------------- STIFFNESS MATRIX --------------------
    // For each tetrahedron
    for (int t = 0; t < T.rows(); t++) {
        // Get vertex indices
        int v0 = T(t, 0);
        int v1 = T(t, 1);
        int v2 = T(t, 2);
        int v3 = T(t, 3);

        // Get vertex positions
        Vector3d x0 = origPositions.segment<3>(3 * v0);
        Vector3d x1 = origPositions.segment<3>(3 * v1);
        Vector3d x2 = origPositions.segment<3>(3 * v2);
        Vector3d x3 = origPositions.segment<3>(3 * v3);

        // Shape matrix
        Matrix3d Dm;
        Dm.col(0) = x1 - x0;
        Dm.col(1) = x2 - x0;
        Dm.col(2) = x3 - x0;

        // Inverse shape matrix
        Matrix3d DmInv = Dm.inverse();

        // Tetrahedron volume
        double vol = std::abs(Dm.determinant()) / 6.0;

        // Shape function gradients
        std::vector<Vector3d> gradients(4);
        gradients[1] = DmInv.col(0);
        gradients[2] = DmInv.col(1);
        gradients[3] = DmInv.col(2);
        gradients[0] = -(gradients[1] + gradients[2] + gradients[3]);

        // Array of vertex indices
        int vertexIndices[4] = {v0, v1, v2, v3};

        // For each vertex pair
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                // 3x3 block for this vertex pair
                Matrix3d Kij = Matrix3d::Zero();

                // Lambda contribution (isotropic)
                Kij += lambda * gradients[i].dot(gradients[j]) * Matrix3d::Identity();

                // Mu contribution (shear)
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        Kij(a, b) += mu * (gradients[i](b) * gradients[j](a) +
                                          gradients[i](a) * gradients[j](b));
                    }
                }

                // Scale by volume
                Kij *= vol;

                // Add to triplets
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        KTriplets.push_back(Triplet<double>(
                            3 * vertexIndices[i] + a,
                            3 * vertexIndices[j] + b,
                            Kij(a, b)
                        ));
                    }
                }
            }
        }
    }

    // Create sparse mass and stiffness matrices
    M.resize(dim, dim);
    K.resize(dim, dim);
    M.setFromTriplets(MTriplets.begin(), MTriplets.end());
    K.setFromTriplets(KTriplets.begin(), KTriplets.end());

    // -------------------- DAMPING MATRIX --------------------
    // Create Rayleigh damping matrix: D = alpha*M + beta*K
    D = alpha * M + beta * K;
    }
    
    //returns center of mass
    Vector3d initializeVolumesAndMasses()
    {
        //TODO: compute tet volumes and allocate to vertices
        tetVolumes.conservativeResize(T.rows());
        voronoiVolumes.conservativeResize(origPositions.size()/3);
        voronoiVolumes.setZero();
        invMasses.conservativeResize(origPositions.size()/3);
        Vector3d COM; COM.setZero();
        for (int i=0;i<T.rows();i++){
            Vector3d e01=origPositions.segment(3*T(i,1),3)-origPositions.segment(3*T(i,0),3);
            Vector3d e02=origPositions.segment(3*T(i,2),3)-origPositions.segment(3*T(i,0),3);
            Vector3d e03=origPositions.segment(3*T(i,3),3)-origPositions.segment(3*T(i,0),3);
            Vector3d tetCentroid=(origPositions.segment(3*T(i,0),3)+origPositions.segment(3*T(i,1),3)+origPositions.segment(3*T(i,2),3)+origPositions.segment(3*T(i,3),3))/4.0;
            tetVolumes(i)=std::abs(e01.dot(e02.cross(e03)))/6.0;
            for (int j=0;j<4;j++)
                voronoiVolumes(T(i,j))+=tetVolumes(i)/4.0;
            
            COM+=tetVolumes(i)*tetCentroid;
        }
        
        COM.array()/=tetVolumes.sum();
        totalInvMass=0.0;
        for (int i=0;i<origPositions.size()/3;i++){
            invMasses(i)=1.0/(voronoiVolumes(i)*density);
            totalInvMass+=voronoiVolumes(i)*density;
        }
        totalInvMass = 1.0/totalInvMass;
        
        return COM;
        
    }
    
    Mesh(const VectorXd& _origPositions, const MatrixXi& boundF, const MatrixXi& _T, const int _globalOffset, const double _youngModulus, const double _poissonRatio, const double _density, const bool _isFixed, const RowVector3d& userCOM, const RowVector4d& userOrientation){
        origPositions=_origPositions;
        //cout<<"original origPositions: "<<origPositions<<endl;
        T=_T;
        F=boundF;
        isFixed=_isFixed;
        globalOffset=_globalOffset;
        density=_density;
        poissonRatio=_poissonRatio;
        youngModulus=_youngModulus;
        currVelocities=VectorXd::Zero(origPositions.rows());
        
        VectorXd naturalCOM=initializeVolumesAndMasses();
        //cout<<"naturalCOM: "<<naturalCOM<<endl;
        
        
        origPositions-= naturalCOM.replicate(origPositions.rows()/3,1);  //removing the natural COM of the OFF file (natural COM is never used again)
        //cout<<"after natrualCOM origPositions: "<<origPositions<<endl;
        
        for (int i=0;i<origPositions.size();i+=3)
            origPositions.segment(i,3)<<(QRot(origPositions.segment(i,3).transpose(), userOrientation)+userCOM).transpose();
        
        currPositions=origPositions;
        
        if (isFixed)
            invMasses.setZero();
        
        //finding boundary tets
        VectorXi boundVMask(origPositions.rows()/3);
        boundVMask.setZero();
        for (int i=0;i<boundF.rows();i++)
            for (int j=0;j<3;j++)
                boundVMask(boundF(i,j))=1;
        
        //cout<<"boundVMask.sum(): "<<boundVMask.sum()<<endl;
        
        vector<int> boundTList;
        for (int i=0;i<T.rows();i++){
            int incidence=0;
            for (int j=0;j<4;j++)
                incidence+=boundVMask(T(i,j));
            if (incidence>2)
                boundTList.push_back(i);
        }
        
        boundTets.resize(boundTList.size());
        for (int i=0;i<boundTets.size();i++)
            boundTets(i)=boundTList[i];
        
    }
    
};





#endif
