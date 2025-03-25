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
        
    alpha = _alpha;
    beta = _beta;

    // Compute the Lamé parameters from Young's modulus and Poisson ratio
    double mu = youngModulus / (2.0 * (1.0 + poissonRatio));
    double lambda = youngModulus * poissonRatio / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));

    // Resize matrices to 3|V| × 3|V|
    int n = origPositions.size();
    K.resize(n, n);
    M.resize(n, n);
    D.resize(n, n);

    // Initialize with triplets for sparse matrix construction
    std::vector<Eigen::Triplet<double>> KTriplets;
    std::vector<Eigen::Triplet<double>> MTriplets;

    // Build the mass matrix (lumped mass matrix)
    for (int i = 0; i < invMasses.size(); i++) {
        double mass = 1.0 / invMasses(i);
        for (int j = 0; j < 3; j++) {
            MTriplets.push_back(Eigen::Triplet<double>(3*i+j, 3*i+j, mass));
        }
    }

    // Construct stiffness matrix based on each tetrahedron
    for (int t = 0; t < T.rows(); t++) {
        // Get the indices of the four vertices of the tetrahedron
        int v0 = T(t, 0);
        int v1 = T(t, 1);
        int v2 = T(t, 2);
        int v3 = T(t, 3);

        // Get the original positions of the four vertices
        Vector3d x0 = origPositions.segment<3>(3*v0);
        Vector3d x1 = origPositions.segment<3>(3*v1);
        Vector3d x2 = origPositions.segment<3>(3*v2);
        Vector3d x3 = origPositions.segment<3>(3*v3);

        // Compute the shape matrix Dm (matrix of edge vectors)
        Matrix3d Dm;
        Dm.col(0) = x1 - x0;
        Dm.col(1) = x2 - x0;
        Dm.col(2) = x3 - x0;

        // Compute the inverse of the shape matrix
        Matrix3d DmInv = Dm.inverse();

        // Compute shape functions gradients (which are constant per tetrahedron)
        Vector3d grad0 = -DmInv.col(0) - DmInv.col(1) - DmInv.col(2);
        Vector3d grad1 = DmInv.col(0);
        Vector3d grad2 = DmInv.col(1);
        Vector3d grad3 = DmInv.col(2);

        // Volume of the tetrahedron (already computed in tetVolumes)
        double volume = tetVolumes(t);

        // For each pair of vertices in the tetrahedron
        std::vector<Vector3d> gradients = {grad0, grad1, grad2, grad3};
        std::vector<int> indices = {v0, v1, v2, v3};

        // Build the element stiffness matrix
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                // Compute the 3x3 block contribution
                Matrix3d Kij = Matrix3d::Zero();

                // Contribution from first Lamé parameter (lambda)
                Kij += lambda * gradients[i].dot(gradients[j]) * Matrix3d::Identity();

                // Contribution from second Lamé parameter (mu)
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        Kij(a, b) += mu * (gradients[i](b) * gradients[j](a) + gradients[i](a) * gradients[j](b));
                    }
                }

                // Scale by volume
                Kij *= volume;

                // Add to global stiffness matrix
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        KTriplets.push_back(Eigen::Triplet<double>(3*indices[i]+a, 3*indices[j]+b, Kij(a, b)));
                    }
                }
            }
        }
    }

    // Set matrices from triplets
    K.setFromTriplets(KTriplets.begin(), KTriplets.end());
    M.setFromTriplets(MTriplets.begin(), MTriplets.end());

    // Compute damping matrix using Rayleigh damping (D = alpha*M + beta*K)
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
