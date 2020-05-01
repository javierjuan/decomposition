/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2014 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Non-Negative Matrix Factorization                                        *
***************************************************************************/

#ifndef NNMF_HPP
#define NNMF_HPP

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <algorithm>

#define SQRTEPS 1.490116119384766e-08

// [1] C. Boutsidis, E. Gallopoulos. SVD based initialization:A head start for nonnegative matrix factorization. Pattern Recognition 41 (2008) 1350–1362

using namespace Eigen;

class NNMF
{
public:
	enum Algorithm { ALS, MUR };

	NNMF(const int maxIterations = 200, const double threshold = 1e-4, const double tolerance = 1e-4, const bool verbose = false);
	
	template <class Derived>
	void compute(const MatrixBase<Derived> &dataset, const int K, const Algorithm algorithm = ALS);
	template <class Derived>
	void compute(const MatrixBase<Derived> &dataset, const MatrixXd &W0, const MatrixXd &H0, const Algorithm algorithm = ALS);
	
	MatrixXd W() const { return m_W; }
	MatrixXd H() const { return m_H; }
	
private:
	template <bool AlternatingLeastSquares = true>
	void nnmf(const MatrixXd &dataset);
	
	template <class Derived>
	void SVDInitialization(const MatrixBase<Derived> &dataset, const int K);
	void standard_form();
	
	MatrixXd m_W;
	MatrixXd m_H;
	int m_components;
	int m_maxIterations;
	double m_threshold;
	double m_tolerance;
	bool m_verbose;
};

/***************************** Implementation *****************************/

NNMF::NNMF(const int maxIterations, const double threshold, const double tolerance, const bool verbose)
: m_W(MatrixXd()), m_H(MatrixXd()), m_components(0), m_maxIterations(maxIterations), m_threshold(threshold), m_tolerance(tolerance), m_verbose(verbose)
{
}

template<class Derived>
void NNMF::compute(const MatrixBase<Derived> &dataset, const int K, const Algorithm algorithm)
{
	if (K >= dataset.cols())
	{
		m_W = dataset.derived().cast<double>();
		m_H = MatrixXd::Identity(K, K);
		return;
	}

	if (K >= dataset.rows())
	{
		m_W = MatrixXd::Identity(K, K);
		m_H = dataset.derived().cast<double>();
		return;
	}
	
	// Set components
	m_components = K;
	
	// Singular Value Decomposition initialization
	SVDInitialization(dataset, K);

	// Run Non-Negative Matrix Factorization decomposition
	if (algorithm == ALS)
		nnmf<true>(dataset.derived().cast<double>());
	else if (algorithm == MUR)
		nnmf<false>(dataset.derived().cast<double>());
	else
		nnmf<true>(dataset.derived().cast<double>());
}

template<class Derived>
void NNMF::compute(const MatrixBase<Derived> &dataset, const MatrixXd &W0, const MatrixXd &H0, const Algorithm algorithm)
{
	// Set components
	m_components = W0.cols();
	
	// Initialize W and H matrices
	m_W = W0;
	m_H = H0;

	// Run Non-Negative Matrix Factorization decomposition
	if (algorithm == ALS)
		nnmf<true>(dataset.derived().cast<double>());
	else if (algorithm == MUR)
		nnmf<false>(dataset.derived().cast<double>());
	else
		nnmf<true>(dataset.derived().cast<double>());
}

template <bool AlternatingLeastSquares>
void NNMF::nnmf(const MatrixXd &dataset)
{
	// Set number of elements
	const double NM = (double) dataset.size();

	// Declare previous iteration matrices and error
	MatrixXd W0 = m_W;
	MatrixXd H0 = m_H;
	double dnorm0 = 0;
	
	if (m_verbose)
	{
		mexPrintf("Iteration\t\tMax iterations\t\tRMSE\t\t\tRatio\t\t\tThreshold\t\tDelta\t\t\tTolerance\n");
		mexPrintf("-------------------------------------------------------------------------------------------------------------\n");
	}
	
	// Constant matrices
	const MatrixXd H_MUR_SQRTEPS = MatrixXd::Constant(m_components, dataset.cols(), SQRTEPS);
	const MatrixXd W_MUR_SQRTEPS = MatrixXd::Constant(dataset.rows(), m_components, SQRTEPS);
	
	for (int i = 0; i < m_maxIterations; ++i)
	{
		if (AlternatingLeastSquares)
		{
			// Update H
			m_H = W0.fullPivHouseholderQr().solve(dataset).cwiseMax(0);
			// Update W
			m_W = m_H.transpose().fullPivHouseholderQr().solve(dataset.transpose()).transpose().cwiseMax(0);
		}
		else
		{
			// Update H
			MatrixXd numer = W0.transpose() * dataset;
			m_H = H0.cwiseProduct(numer.cwiseQuotient(((W0.transpose() * W0) * H0) + H_MUR_SQRTEPS)).cwiseMax(0);
			// Update W
			numer = dataset * m_H.transpose();
			m_W = W0.cwiseProduct(numer.cwiseQuotient((W0 * (m_H * m_H)) + W_MUR_SQRTEPS)).cwiseMax(0);
		}
		
		// Compute squared error
		const ArrayXXd d = dataset - (m_W * m_H);
		const double dnorm = std::sqrt((d * d).sum() / NM);		
		const double dW = ((m_W - W0).cwiseAbs() / (SQRTEPS + W0.cwiseAbs().maxCoeff())).maxCoeff();
		const double dH = ((m_H - H0).cwiseAbs() / (SQRTEPS + H0.cwiseAbs().maxCoeff())).maxCoeff();
		const double delta = std::max(dW, dH);
		
		if (m_verbose)
		{
			mexPrintf("%d\t\t\t\t%d\t\t\t\t\t%.2e\t\t%.2e\t\t%.2e\t\t%.2e\t\t%.2e\n", i+1, m_maxIterations, dnorm, (dnorm0 - dnorm), (m_threshold * std::max(1.0, dnorm0)), delta, m_tolerance);
			mexEvalString("drawnow");
		}
		
		if (i > 0)
		{
			if (delta <= m_tolerance)
			{
				standard_form();
				return;
			}
			if (dnorm0 - dnorm <= (m_threshold * std::max(1.0, dnorm0)))
			{
				standard_form();
				return;
			}
		}
		
		dnorm0 = dnorm;
		W0 = m_W;
		H0 = m_H;
	}
}

template <class Derived>
void NNMF::SVDInitialization(const MatrixBase<Derived> &dataset, const int K)
{
	// Allocate memory
	m_W.resize(dataset.rows(), K);
	m_H.resize(K, dataset.cols());
	
	// Perform SVD decomposition
	JacobiSVD<MatrixXd> svd(dataset, ComputeThinU | ComputeThinV);
	MatrixXd U = svd.matrixU();
	MatrixXd V = svd.matrixV();
	VectorXd S = svd.singularValues();
	
	// Retain only the largest K eigen values and eigen vectors
	S = S.segment(0, K).eval();
	U = U.block(0, 0, U.rows(), K).eval();
	V = V.block(0, 0, V.rows(), K).eval();
	
	// Perform initialization as suggested in [1]
	m_W.col(0) = std::sqrt(S(0)) * U.col(0);
	m_H.row(0) = std::sqrt(S(0)) * V.col(0).transpose();
	
	for (int i = 1; i < K; ++i)
	{
		const VectorXd Uip = U.col(i).cwiseMax(0);
		const VectorXd Uin = U.col(i).cwiseMin(0).cwiseAbs();
		
		const VectorXd Vip = V.col(i).cwiseMax(0);
		const VectorXd Vin = V.col(i).cwiseMin(0).cwiseAbs();
		
		const double UipNorm = Uip.norm();
		const double VipNorm = Vip.norm();
		
		const double UinNorm = Uin.norm();
		const double VinNorm = Vin.norm();
		
		const double mp = UipNorm * VipNorm;
		const double mn = UinNorm * VinNorm;
		
		VectorXd u;
		VectorXd v;
		double sigma;
		if (mp > mn)
		{
			u = Uip / UipNorm;
			v = Vip / VipNorm;
			sigma = mp;			
		}
		else
		{
			u = Uin / UinNorm;
			v = Vin / VinNorm;
			sigma = mn;			
		}
		m_W.col(i) = std::sqrt(S(i) * sigma) * u;
		m_H.row(i) = std::sqrt(S(i) * sigma) * v.transpose();
	}
}

void NNMF::standard_form()
{
	// Put in standard form
	RowVectorXd Hlen = m_H.cwiseProduct(m_H).rowwise().sum().cwiseSqrt();
	
	for (int i = 0; i < Hlen.size(); ++i)
		Hlen(i) = Hlen(i) == 0 ? 1.0 : Hlen(i);
	
	for (int i = 0; i < m_W.rows(); ++i)
		m_W.row(i) = m_W.row(i).cwiseProduct(Hlen);
	
	for (int i = 0; i < m_H.cols(); ++i)
		m_H.col(i) = m_H.col(i).cwiseQuotient(Hlen.transpose());
	
	const RowVectorXd magnitude = m_W.cwiseProduct(m_W).colwise().sum();
	std::vector<int> y(magnitude.size());
    int n = 0;
    std::generate(std::begin(y), std::end(y), [&] { return n++; });
	std::sort(std::begin(y), std::end(y), [&](int i1, int i2) { return magnitude(i1) > magnitude(i2); });
	
	const MatrixXd Wcopy = m_W;
	const MatrixXd Hcopy = m_H;
	for (int i = 0; i < magnitude.size(); ++i)
	{
		m_W.col(i) = Wcopy.col(y[i]);
		m_H.row(i) = Hcopy.row(y[i]);
	}
}

#endif