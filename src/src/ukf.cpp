#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
	x_.fill(0.0);
  // initial covariance matrix
  P_ = MatrixXd(5, 5);
	P_.fill(0.0);
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.8;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.8;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  n_x_ = 5;

  n_aug_ = 7;

  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  weights_ = VectorXd(2 * n_aug_ + 1);

  time_us_ = 0;

  weights_ = VectorXd(2 * n_aug_ + 1);
  double weight = 0.5 / (lambda_ + n_aug_);
  weights_[0] = lambda_ / (lambda_ + n_aug_);
  for (size_t i = 1; i < 2 * n_aug_; i++)
  {
	weights_(i) = weight;
  }

  n_z = 3;

  Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
   
  is_initialized_ = false;
  
  NIS_radar_ = 0;
  NIS_laser_ = 0;
}

UKF::~UKF() {}




void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	/**
	 * TODO: Complete this function! Make sure you switch between lidar and radar
	 * measurements.
	 */
	if (!is_initialized_) {

		if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			x_ << meas_package.raw_measurements_[0],
				meas_package.raw_measurements_[1],
				0,
				0,
				0;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			x_ << (cos(meas_package.raw_measurements_[1]) * meas_package.raw_measurements_[0]),
				(sin(meas_package.raw_measurements_[1]) * meas_package.raw_measurements_[0]),
				(sqrt((pow(cos(meas_package.raw_measurements_[1]) * meas_package.raw_measurements_[2], 2)) + pow(sin(meas_package.raw_measurements_[1]) * meas_package.raw_measurements_[2], 2))),
				0,
				0;
		}
		time_us_ = meas_package.timestamp_;
		P_ << 0.3, 0, 0, 0, 0,
			0, 0.3, 0, 0, 0,
			0, 0, 0.3, 0, 0,
			0, 0, 0, 0.3, 0,
			0, 0, 0, 0, 0.3;

		is_initialized_ = true;
		return;
	}

	float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
	time_us_ = meas_package.timestamp_;

	Prediction(dt);
	if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
		UpdateLidar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
		UpdateRadar(meas_package);
	}
}

void UKF::Prediction(double delta_t) {

	// create sigma point matrix
	VectorXd x_aug = VectorXd(7);
	x_aug.fill(0.0);
	// create augmented state covariance
	MatrixXd P_aug = MatrixXd(7, 7);

	// create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	Xsig_aug.fill(0.0);
	// create augmented mean state
	x_aug.head(n_x_) = x_;
	x_aug[5] = 0;
	x_aug[6] = 0;
	// create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(5, 5) = pow(std_a_, 2);
	P_aug(6, 6) = pow(std_yawdd_, 2);
	// create square root matrix
	MatrixXd A = P_aug.llt().matrixL();
	// create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (size_t i = 1; i <= n_aug_; i++)
	{
		Xsig_aug.col(i) = x_aug + sqrtf(lambda_ + n_aug_) * A.col(i - 1);
		Xsig_aug.col(i + n_aug_) = x_aug - sqrtf(lambda_ + n_aug_) * A.col(i - 1);
	}

	VectorXd Xf = VectorXd(n_x_);
	VectorXd X_vk = VectorXd(n_x_);
	Xf.fill(0.0);
	X_vk.fill(0.0);
	// predict sigma points
	for (size_t i = 0; i < 2 * n_aug_ + 1; i++)
	{
		X_vk << 0.5 * pow(delta_t, 2) * cos(Xsig_aug(3, i)) * Xsig_aug(5, i),
			0.5 * pow(delta_t, 2) * sin(Xsig_aug(3, i)) * Xsig_aug(5, i),
			delta_t* Xsig_aug(5, i),
			0.5 * pow(delta_t, 2) * Xsig_aug(6, i),
			delta_t* Xsig_aug(6, i);

		if (fabs(Xsig_aug(4, i)) > 0.001) {
			Xf << (Xsig_aug(2, i) / Xsig_aug(4, i)) * ((sin(Xsig_aug(3, i) + Xsig_aug(4, i) * delta_t)) - (sin(Xsig_aug(3, i)))),
				(Xsig_aug(2, i) / Xsig_aug(4, i))* ((-cos(Xsig_aug(3, i) + Xsig_aug(4, i) * delta_t)) + (cos(Xsig_aug(3, i)))),
				0,
				Xsig_aug(4, i)* delta_t,
				0;
		}
		else {
			Xf << Xsig_aug(2, i) * cosf(Xsig_aug(3, i)) * delta_t,
				Xsig_aug(2, i)* sinf(Xsig_aug(3, i))* delta_t,
				0,
				0,
				0;
		}
		Xsig_pred.col(i) = Xsig_aug.topLeftCorner(5, 15).col(i) + Xf + X_vk;
	}
	x_.fill(0.0);
	for (size_t i = 0; i < 2 * n_aug_ + 1; i++)
	{
		x_ = x_ + (weights_[i] * Xsig_pred.col(i));
	}
	P_.fill(0.0);
	for (size_t i = 0; i < 2 * n_aug_ + 1; i++)
	{
		P_ = P_ + (weights_[i] * (Xsig_pred.col(i) - x_) * (Xsig_pred.col(i) - x_).transpose());
	}
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {

	int n_z_lidar = 2;
	MatrixXd Zsig = MatrixXd(n_z_lidar, 2 * n_aug_ + 1);
	VectorXd z_pred = VectorXd(n_z_lidar);
	MatrixXd S = MatrixXd(n_z_lidar, n_z_lidar);
	MatrixXd R = MatrixXd(n_z_lidar, n_z_lidar);
	R << std_laspx_ * std_laspx_, 0,
		0, std_laspy_* std_laspy_;
	Zsig.fill(0.0);
	for (size_t i = 0; i < 2 * n_aug_ + 1; i++) {
		Zsig.col(i)[0] = Xsig_pred.col(i)[0];
		Zsig.col(i)[1] = Xsig_pred.col(i)[1];
	}
	z_pred(0.0);
	for (size_t i = 0; i < 2 * n_aug_ + 1; i++)
	{
		z_pred = z_pred + (weights_[i] * Zsig.col(i));
	}
	S.fill(0.0);
	for (size_t i = 0; i < 2 * n_aug_ + 1; i++)
	{
		S = S + (weights_[i] * (Zsig.col(i) - z_pred) * (Zsig.col(i) - z_pred).transpose());
	}
	S = S + R;
	VectorXd z = VectorXd(n_z_lidar);
	z << meas_package.raw_measurements_;
	MatrixXd Tc = MatrixXd(n_x_, n_z_lidar);
	MatrixXd K = MatrixXd(n_x_, n_z_lidar);
	Tc.fill(0.0);
	// calculate cross correlation matrix
	for (size_t i = 0; i < 2 * n_aug_ + 1; i++)
	{
		Tc = Tc + (weights_[i] * (Xsig_pred.col(i) - x_) * (Zsig.col(i) - z_pred).transpose());
	}
	K.fill(0.0);
	// calculate Kalman gain K;
	K = Tc * (S.inverse());
	// update state mean and covariance matrix
	x_ = x_ + (K * (z - z_pred));
	P_ = P_ - (K * S * K.transpose());
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {

	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
	VectorXd z_pred = VectorXd(n_z);
	MatrixXd S = MatrixXd(n_z, n_z);
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_radr_ * std_radr_, 0, 0,
		0, std_radphi_* std_radphi_, 0,
		0, 0, std_radrd_* std_radrd_;

	for (size_t i = 0; i < 2 * n_aug_ + 1; i++)
	{
		Zsig.col(i)[0] = sqrtf(pow(Xsig_pred.col(i)[0], 2) + pow(Xsig_pred.col(i)[1], 2));
		Zsig.col(i)[1] = atan2(Xsig_pred.col(i)[1], Xsig_pred.col(i)[0]);
		Zsig.col(i)[2] = ((Xsig_pred.col(i)[0] * cos(Xsig_pred.col(i)[3]) * Xsig_pred.col(i)[2]) + (Xsig_pred.col(i)[1] * sin(Xsig_pred.col(i)[3]) * Xsig_pred.col(i)[2])) / (Zsig.col(i)[0]);
	}
	z_pred.fill(0.0);
	for (size_t i = 0; i < 2 * n_aug_ + 1; i++)
	{
		z_pred = z_pred + (weights_[i] * Zsig.col(i));
	}
	S.fill(0.0);
	for (size_t i = 0; i < 2 * n_aug_ + 1; i++)
	{
		S = S + (weights_[i] * (Zsig.col(i) - z_pred) * (Zsig.col(i) - z_pred).transpose());
	}
	S = S + R;

	// create example vector for incoming radar measurement
	VectorXd z = VectorXd(n_z);
	z << meas_package.raw_measurements_;

	// create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);
	MatrixXd K = MatrixXd(n_x_, n_z);

	Tc.fill(0.0);
	// calculate cross correlation matrix
	for (size_t i = 0; i < 2 * n_aug_ + 1; i++)
	{
		Tc = Tc + (weights_[i] * (Xsig_pred.col(i) - x_) * (Zsig.col(i) - z_pred).transpose());
	}
	K.fill(0.0);
	// calculate Kalman gain K;
	K = Tc * S.inverse();
	// update state mean and covariance matrix
	x_ = x_ + (K * (z - z_pred));
	P_ = P_ - (K * S * K.transpose());
}