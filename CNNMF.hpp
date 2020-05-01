/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2014 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Convex Non-Negative Matrix Factorization                                 *
***************************************************************************/

#ifndef CNNMF_HPP
#define CNNMF_HPP

#include <Eigen/Dense>
#include <cmath>
#include "NNMF.hpp"

// [1] C. Ding and T. Li and M. Jordan, "Convex and Semi-Nonnegative Matrix
//     Factorizations", IEEE Transactions on Pattern Analysis and Machine
//     Intelligence, Vol. 99(1), 2008.

using namespace Eigen;

class CNNMF
{
public:
	CNNMF(const int maxIterations = 200, const double threshold = 1e-4, const double tolerance = 1e-4, const bool verbose = false);
	
	template<class Derived>
	void compute(const MatrixBase<Derived> &dataset, const int K);
	template<class Derived>
	void compute(const MatrixBase<Derived> &dataset, const MatrixXd &G0, const MatrixXd &H0);
	
	MatrixXd G() const { return m_G; }
	MatrixXd H() const { return m_H; }
	
private:
	void cnnmf_mur(const MatrixXd &dataset);
	
	MatrixXd m_G;
	MatrixXd m_H;
	int m_components;
	int m_maxIterations;
	double m_threshold;
	double m_tolerance;
	bool m_verbose;
};

/***************************** Implementation *****************************/

CNNMF::CNNMF(const int maxIterations, const double threshold, const double tolerance, const bool verbose)
: m_G(MatrixXd()), m_H(MatrixXd()), m_components(0), m_maxIterations(maxIterations), m_threshold(threshold), m_tolerance(tolerance), m_verbose(verbose)
{
}

template<class Derived>
void CNNMF::compute(const MatrixBase<Derived> &dataset, const int K)
{
	if (K >= dataset.cols() || K >= dataset.rows())
	{
		m_G = MatrixXd::Identity(dataset.cols(), K);
		m_H = MatrixXd::Identity(K, dataset.cols());
		return;
	}
	
	// Initialize G and H matrices
	NNMF nnmf(m_maxIterations, m_threshold, m_tolerance, true);
	nnmf.compute(dataset, K);
	
	m_H = nnmf.H();
	m_G = m_H.transpose().fullPivHouseholderQr().solve(MatrixXd::Identity(dataset.cols(), dataset.cols())).transpose().cwiseMax(0);
	
	// Set components
	m_components = K;
	
	// Compute Convex Non-Negative Matrix Factorization decomposition
	cnnmf_mur(dataset.cast<double>());
}

template<typename T>
void CNNMF::compute(const MatrixBase<T> &dataset, const MatrixXd &G0, const MatrixXd &H0)
{
	// Initialize W and H matrices
	m_components = G0.cols();
	m_G = G0;
	m_H = H0;
	
	// Compute Convex Non-Negative Matrix Factorization decomposition
	cnnmf_mur(dataset.cast<double>());
}

// Multiplicative Updated Rules
void CNNMF::cnnmf_mur(const MatrixXd &dataset)
{
	// Set dimension variables
	const double NM = (double) (dataset.rows() * dataset.cols());

	// Set usefull matrices	
	const MatrixXd VtV = dataset.transpose() * dataset;
	const MatrixXd Yp = (VtV.cwiseAbs() + VtV) / 2.0;
	const MatrixXd Yn = (VtV.cwiseAbs() - VtV) / 2.0;
	MatrixXd num, den;
	
	if (m_verbose)
	{
		mexPrintf("Iteration\t\tMax iterations\t\tRMSE\t\t\tRatio\t\tThreshold\n");
		mexPrintf("-----------------------------------------------------------------------------\n");
	}
	
	// Declare previous iteration matrices and error
	double dnorm0 = 0;
	
	for (int i = 0; i < m_maxIterations; ++i)
	{
		// Update G
		num = (Yp + (Yn * m_G * m_H)) * m_H.transpose();
		den = (Yn + (Yp * m_G * m_H)) * m_H.transpose();
		m_G = m_G.cwiseProduct(num.cwiseQuotient(den).cwiseSqrt()).cwiseMax(0).eval();
	
		// Update H
		num = m_G.transpose() * (Yp + (Yn * m_G * m_H));
		den = m_G.transpose() * (Yn + (Yp * m_G * m_H));
		m_H = m_H.cwiseProduct(num.cwiseQuotient(den).cwiseSqrt()).cwiseMax(0).eval();
		
		// Compute squared error
		const ArrayXXd d = dataset - (dataset * m_G * m_H.transpose());
		const double dnorm = std::sqrt((d * d).sum() / NM);		
		
		if (m_verbose)
		{
			mexPrintf("%d\t\t\t\t%d\t\t\t\t\t%.2e\t\t%.2e\t\t%.2e\n", i+1, m_maxIterations, dnorm, (dnorm0 - dnorm), (m_threshold * std::max(1.0, dnorm0)));
			mexEvalString("drawnow");
		}
		
		if (i > 0)
		{
			if (dnorm0 - dnorm <= (m_threshold * std::max(1.0, dnorm0)))
				return;
		}
		
		dnorm0 = dnorm;
	}
}

#endif