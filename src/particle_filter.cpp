/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

//using std::string;
//using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  num_particles = 100;  // TODO: Set the number of particles
  
  // Declare a random engine generates pseudo-random numbers
  default_random_engine gen;

  // Creating normal distributions
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Generate particles with normal distribution with mean on GPS values.
  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
  }

  // The filter is now initialized.
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  // Declare a random engine generates pseudo-random numbers
  default_random_engine gen;

  // Calculate new state.
  for (int i = 0; i < num_particles; i++) {
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;
    
    double pred_x;
    double pred_y;
    double pred_theta;

    if ( fabs(yaw_rate) < 0.0001 ) { // When yaw is not changing.
      pred_x = particle_x + velocity * cos(particle_theta) * delta_t;
      pred_y = particle_y + velocity * sin(particle_theta) * delta_t;
      pred_theta = particle_theta;
      // yaw continue to be the same.
    } else {
      pred_x = particle_x + (velocity/yaw_rate) * (sin(particle_theta + (yaw_rate * delta_t)) - sin(particle_theta));
      pred_y = particle_y + (velocity/yaw_rate) * (cos(particle_theta) - cos(particle_theta + (yaw_rate * delta_t)));
      pred_theta = particle_theta + (yaw_rate * delta_t);
    }
    
    // Creating normal distributions
    normal_distribution<double> dist_x(pred_x, std_pos[0]);
    normal_distribution<double> dist_y(pred_y, std_pos[1]);
    normal_distribution<double> dist_theta(pred_theta, std_pos[2]);

    // Adding noise.
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
   
  unsigned int nObservations = observations.size();
  unsigned int nPredictions = predicted.size();
  
  for (unsigned int i = 0; i < nObservations; i++) { // For each observation
    
    double minDistance = numeric_limits<double>::max(); // init min distance

    // Initialize the found map in something not possible.
    int mapId = -1;
    
    double obs_x = observations[i].x;
    double obs_y = observations[i].y;

    for (unsigned int j = 0; j < nPredictions; j++ ) { // For each 
      double pred_x = predicted[j].x;
      double pred_y = predicted[j].y;
      double distance = dist(obs_x, obs_y, pred_x, pred_y);

      // If the "distance" is less than min, stored the id and update min.
      if ( distance < minDistance ) {
        minDistance = distance;
        mapId = predicted[j].id;
      }
    }

    // Update the observation identifier.
    observations[i].id = mapId;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  double weight_normalizer = 0.0; // To normalize weights of all particles

  for (int i = 0; i < num_particles; i++) {
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;

    /*Step 1: Transform observations from vehicle co-ordinates to map co-ordinates.*/
    vector<LandmarkObs> transformed_obs;
	unsigned int nObservations = observations.size();
    for (unsigned int j = 0; j < nObservations; j++) {
      LandmarkObs transformed_obs;
      transformed_obs.id = j;
      transformed_obs.x = particle_x + (cos(particle_theta) * observations[j].x) - (sin(particle_theta) * observations[j].y);
      transformed_obs.y = particle_y + (sin(particle_theta) * observations[j].x) + (cos(particle_theta) * observations[j].y);
      transformed_obs.push_back(transformed_obs);
    }

    /*Step 2: Keep landmarks which are in the sensor_range of current particle.*/
    vector<LandmarkObs> predicted_landmarks;
	unsigned int nLandmarks = map_landmarks.landmark_list.size();
    for (unsigned int j = 0; j < nLandmarks; j++) {
		int landmark_id = map_landmarks.landmark_list[j].id_i;
		double landmark_x = map_landmarks.landmark_list[j].x_f;
	    double landmark_y = map_landmarks.landmark_list[j].y_f;
		
		double landmark_distance = dist(particle_x , particle_y, landmark_x, landmark_y);
		
		if (landmark_distance <= sensor_range) {
	    	predicted_landmarks.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
	    }
    }

    /*Step 3: Associate observations with predicted landmarks */
    dataAssociation(predicted_landmarks, transformed_obs);

    /*Step 4: Update weight of each particle using Multivariate Gaussian distribution.*/
    particles[i].weight = 1.0; // Reset the weight of particle to 1.0

    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
    double sigma_x_2 = pow(sigma_x, 2);
    double sigma_y_2 = pow(sigma_y, 2);
    double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));
    
    //Calculate weight of each particle
	unsigned int nTransformedObs = transformed_obs.size();
	unsigned int nPredLandmarks = predicted_landmarks.size();
    for (unsigned int k = 0; k < nTransformedObs; k++) {
      double trans_obs_x = transformed_obs[k].x;
      double trans_obs_y = transformed_obs[k].y;
      double trans_obs_id = transformed_obs[k].id;
      double multi_prob = 1.0;

      for (unsigned int l = 0; l < nPredLandmarks; l++) {
        double pred_landmark_x = predicted_landmarks[l].x;
        double pred_landmark_y = predicted_landmarks[l].y;
        double pred_landmark_id = predicted_landmarks[l].id;

        if (trans_obs_id == pred_landmark_id) {
          multi_prob = normalizer * exp(-1.0 * ((pow((trans_obs_x - pred_landmark_x), 2)/(2.0 * sigma_x_2)) + 
                                                (pow((trans_obs_y - pred_landmark_y), 2)/(2.0 * sigma_y_2))));
          particles[i].weight *= multi_prob;
        }
      }
    }
    weight_normalizer += particles[i].weight;
  }

  /*Step 5: Normalize the weights of all particles */
  unsigned int nParticles = particles.size();
  for (unsigned int i = 0; i < nParticles; i++) {
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // Create a generator to be used for generating random particle index and beta value
  default_random_engine gen;
  
  // Generate random particle index
  uniform_int_distribution<int> particle_index(0, num_particles - 1);
  int current_index = particle_index(gen);
  
  double beta = 0.0;
  
  double max_weight_2 = 2.0 * *max_element(weights.begin(), weights.end());
  
  vector<Particle> resampled_particles;
  
  for (int i = 0; i < particles.size(); i++) {
    uniform_real_distribution<double> random_weight(0.0, max_weight_2);
    beta += random_weight(gen);
    
    while (beta > weights[current_index]) {
      beta -= weights[current_index];
      current_index = (current_index + 1) % num_particles;
    }
    
    resampled_particles.push_back(particles[current_index]);
  }
  
  particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
  
  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}