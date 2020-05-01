/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2014 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Principal Component Analysis                                             *
***************************************************************************/

#ifndef PRINCIPALCOMPONENTANALYSIS_HPP
#define PRINCIPALCOMPONENTANALYSIS_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <omp.h>

using namespace Eigen;

class PrincipalComponentAnalysis
{
public:
    PrincipalComponentAnalysis();
    
    template <class Derived>
    void compute(const MatrixBase<Derived> &dataset, const bool zscores = false);
    template <class Derived>
    MatrixXd dimensionalityReductionComponents(const MatrixBase<Derived> &dataset, const unsigned int components);
    template <class Derived>
    MatrixXd dimensionalityReductionVarianceExplained(const MatrixBase<Derived> &dataset, const double varianceExplained, const unsigned int minComponents = 0, const unsigned int maxComponents = 0);
    template <class Derived>
    Matrix<typename Derived::Scalar, Dynamic, Dynamic> filteringComponents(const MatrixBase<Derived> &dataset, const unsigned int components);
    template <class Derived>
    Matrix<typename Derived::Scalar, Dynamic, Dynamic> filteringVarianceExplained(const MatrixBase<Derived> &dataset, const double varianceExplained, const unsigned int minComponents = 0, const unsigned int maxComponents = 0);
    
    MatrixXd scores() const { return m_scores; }
    MatrixXd coefficients() const { return m_coefficients; }
    VectorXd latent() const { return m_latent; }
    VectorXd explained() const { return m_explained; }
    RowVectorXd mu() const { return m_mu; } 
    unsigned int components() const { return m_components; }
    
private:
    MatrixXd m_scores;
    MatrixXd m_coefficients;
    VectorXd m_latent;
    VectorXd m_explained;
    RowVectorXd m_mu;
    unsigned int m_components;
    SelfAdjointEigenSolver<MatrixXd> m_solver;
};

/***************************** Implementation *****************************/

PrincipalComponentAnalysis::PrincipalComponentAnalysis()
: m_scores(MatrixXd()), m_coefficients(MatrixXd()), m_latent(VectorXd()), m_explained(VectorXd()), m_mu(RowVectorXd()), m_components(0)
{
}

template<class Derived>
void PrincipalComponentAnalysis::compute(const MatrixBase<Derived> &dataset, const bool zscores)
{
    // Check extreme case
    if (dataset.rows() == 1)    
    {
        m_latent = VectorXd::Ones(1);
        m_explained = m_latent / m_latent.sum();
        m_coefficients = MatrixXd::Ones(dataset.cols(), 1);
        m_scores = dataset.template cast<double>() * m_coefficients;
        return;
    }
    
    // Subtract mean of each variable
    m_mu = dataset.template cast<double>().colwise().mean();
    MatrixXd centered = dataset.template cast<double>().rowwise() - m_mu;
    
    // Standarize std if required
    if (zscores)
    {
        RowVectorXd stdScaling = (centered.cwiseProduct(centered).colwise().sum() / (double) (dataset.rows() - 1)).cwiseSqrt();
        #pragma omp parallel for
        for (int i = 0; i < centered.rows(); ++i)
            centered.row(i) = centered.row(i).cwiseQuotient(stdScaling);
    }
    
    // Compute the covariance matrix.
    MatrixXd covarianceMatrix = (centered.transpose() * centered) / (double)(dataset.rows() - 1);

    // Compute Singular Value Decomposition
    m_solver.compute(covarianceMatrix);
    
    // Store results
    m_latent = m_solver.eigenvalues();
    m_explained = m_latent / m_latent.sum();
    m_coefficients = m_solver.eigenvectors();
    m_scores = centered * m_coefficients;
}

template<class Derived>
MatrixXd PrincipalComponentAnalysis::dimensionalityReductionComponents(const MatrixBase<Derived> &dataset, const unsigned int components)
{
    // Compute PCA
    compute(dataset);
    
    // Get the dataset in the new components dimensional space
    m_components = components;
    return m_scores.rightCols(components);
}

template<typename  Derived>
MatrixXd PrincipalComponentAnalysis::dimensionalityReductionVarianceExplained(const MatrixBase<Derived> &dataset, const double varianceExplained, const unsigned int minComponents, const unsigned int maxComponents)
{
    // Compute PCA
    compute(dataset);
    
    // Get components to explain at least <varianceExplained>
    m_components = 0;
    double cumsum = 0;
    for (int i = m_explained.size() - 1; i >= 0; --i)
    {
        m_components++;
        cumsum += m_explained(i);
        if (cumsum > varianceExplained)
            break;
    }
    if (minComponents != 0)
        m_components = m_components < minComponents ? minComponents : m_components;
    if (maxComponents != 0)
        m_components = m_components > maxComponents ? maxComponents : m_components;
    
    // Get the dataset in the new components dimensional space
    return m_scores.rightCols(m_components);
}

template <typename  Derived>
Matrix<typename Derived::Scalar, Dynamic, Dynamic> PrincipalComponentAnalysis::filteringComponents(const MatrixBase<Derived> &dataset, const unsigned int components)
{
    // Compute PCA
    compute(dataset);

    // Rebuild dataset with components
    m_components = components;
    MatrixXd filteredDataset = (m_scores.rightCols(m_components) * m_coefficients.rightCols(m_components).transpose());
    filteredDataset = filteredDataset.rowwise() + m_mu;

    return filteredDataset.cast<Derived::Scalar>();
}

template <typename  Derived>
Matrix<typename Derived::Scalar, Dynamic, Dynamic> PrincipalComponentAnalysis::filteringVarianceExplained(const MatrixBase<Derived> &dataset, const double varianceExplained, const unsigned int minComponents, const unsigned int maxComponents)
{
    // Compute PCA
    compute(dataset);

    // Get components to explain at least <varianceExplained>
    m_components = 0;
    double cumsum = 0;
    for (int i = m_explained.size() - 1; i >= 0; i--)
    {
        m_components++;
        cumsum += m_explained(i);
        if (cumsum > varianceExplained)
            break;
    }
    if (minComponents != 0)
        m_components = m_components < minComponents ? minComponents : m_components;
    if (maxComponents != 0)
        m_components = m_components > maxComponents ? maxComponents : m_components;

    // Rebuild dataset with components
    MatrixXd filteredDataset = m_scores.rightCols(m_components) * m_coefficients.rightCols(m_components).transpose();
    filteredDataset = filteredDataset.rowwise() + m_mu;

    return filteredDataset.cast<typename Derived::Scalar>();
}

#endif