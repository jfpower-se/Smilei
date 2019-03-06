#ifndef PATCH1D_H
#define PATCH1D_H

#include "Patch.h"
#include "Field1D.h"
#include "cField1D.h"

class SimWindow;

//! Class Patch : sub MPI domain
//!     Collection of patch = MPI domain
class Patch1D : public Patch
{
public:
    //! Constructor for Patch
    Patch1D( Params &params, SmileiMPI *smpi, DomainDecomposition *domain_decomposition, unsigned int ipatch, unsigned int n_moved );
    //! Cloning Constructor for Patch
    Patch1D( Patch1D *patch, Params &params, SmileiMPI *smpi, DomainDecomposition *domain_decomposition, unsigned int ipatch, unsigned int n_moved, bool with_particles );
    
    void initStep2( Params &params, DomainDecomposition *domain_decomposition ) override;
    
    //! Destructor for Patch
    ~Patch1D() override final;
    
    //! Return the volume (or surface or length depending on simulation dimension)
    //! of one cell at the position of a given particle
    double getCellVolume( Particles *p, unsigned int ipart ) override final
    {
        return cell_volume;
    };
    
    // MPI exchange/sum methods for particles/fields
    //   - fields communication specified per geometry (pure virtual)
    // --------------------------------------------------------------
    
    //! init comm / sum densities
    void initSumField( Field *field, int iDim, SmileiMPI *smpi ) override final;
    void reallyinitSumField( Field *field, int iDim ) override final;
    //! finalize comm / sum densities
    void finalizeSumField( Field *field, int iDim ) override final;
    void reallyfinalizeSumField( Field *field, int iDim ) override final;
    
    //! init comm / exchange fields
    void initExchange( Field *field ) override final;
    //! init comm / exchange complex fields
    void initExchangeComplex( Field *field ) override final;
    //! finalize comm / exchange fields
    void finalizeExchange( Field *field ) override final;
    //! finalize comm / exchange complex fields
    void finalizeExchangeComplex( Field *field ) override final;
    //! init comm / exchange fields in direction iDim only
    void initExchange( Field *field, int iDim, SmileiMPI *smpi ) override final;
    //! init comm / exchange complex fields in direction iDim only
    void initExchangeComplex( Field *field, int iDim, SmileiMPI *smpi ) override final;
    //! finalize comm / exchange fields in direction iDim only
    void finalizeExchange( Field *field, int iDim ) override final;
    //! finalize comm / exchange fields in direction iDim only
    void finalizeExchangeComplex( Field *field, int iDim ) override final;
    
    void exchangeField_movewin( Field* field, int clrw ) override final;
    
    // Create MPI_Datatype to exchange fields
    void createType( Params &params ) override final;
    void createType2( Params &params ) override final;
    void cleanType() override final;
    
    //! MPI_Datatype to exchange [ndims_][iDim=0 prim/dial]
    MPI_Datatype ntypeSum_[2][2];
    //! MPI_Datatype to exchange [ndims_+1][iDim=0 prim/dial]
    //!   - +1 : an additional type to exchange clrw lines
    MPI_Datatype ntype_[2][2];
    
    //! MPI_Datatype to exchange [ndims_][iDim=0 prim/dial]
    MPI_Datatype ntypeSum_complex_[2][2];
    //! MPI_Datatype to exchange [ndims_+1][iDim=0 prim/dial]
    //!   - +1 : an additional type to exchange clrw lines
    MPI_Datatype ntype_complex_[2][2];
    
    
};

#endif
