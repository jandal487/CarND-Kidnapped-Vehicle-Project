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
   
  for (int i = 0; i < num_particles; i++) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
	
    // Step 1: Get landmarks in particle's range.
    vector<LandmarkObs> inRangeLandmarks;
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      float landmarkX = map_landmarks.landmark_list[j].x_f;
      float landmarkY = map_landmarks.landmark_list[j].y_f;
      int id = map_landmarks.landmark_list[j].id_i;
      double dX = fabs(x - landmarkX);
      double dY = fabs(y - landmarkY);
      if ( dX <= sensor_range && dY <= sensor_range ) {
        inRangeLandmarks.push_back(LandmarkObs{ id, landmarkX, landmarkY });
      }
    }

    // Step 2: Transform the coordinates
    vector<LandmarkObs> mapped_obs;
    unsigned int nObservations = observations.size();
    for(unsigned int j = 0; j < nObservations; j++) {
      double xx = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
      double yy = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
      mapped_obs.push_back(LandmarkObs{ observations[j].id, xx, yy });
    }

    // Step 3: Observation association to landmark.
    dataAssociation(inRangeLandmarks, mapped_obs);

    // Step 4: Calculate weights.
    particles[i].weight = 1.0; // reseting the weight
    double s_x = std_landmark[0];
    double s_y = std_landmark[1];
    // Iterate each measurement "taken" by the particle
    for (unsigned int j = 0; j < mapped_obs.size(); j++) {
      double obs_X = mapped_obs[j].x;
      double obs_Y = mapped_obs[j].y;
      double x_l,y_l;
      
      unsigned int nLandmarks = inRangeLandmarks.size();
      for (unsigned int k = 0; k < nLandmarks; k++) {
        if (inRangeLandmarks[k].id == mapped_obs[j].id) {
          x_l = inRangeLandmarks[k].x;
          y_l = inRangeLandmarks[k].y;
        }
      }
  
      //calculate measurement weight
      double normalizer = (1/(2 * M_PI * s_x * s_y));
      double gaussian_term1 = pow(obs_X - x_l,2)/(2*pow(s_x, 2));
      double gaussian_term2 = pow(obs_X - y_l,2)/(2*pow(s_y, 2));
      double meas_w = normalizer * exp( -( gaussian_term1 + gaussian_term2 ) );

      if (meas_w == 0) {
        particles[i].weight *= 0.0001;
      } else {
        particles[i].weight *= meas_w;
      }
    }
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
  
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
      weights.push_back(particles[i].weight);
  }
  double max_weight = *max_element(weights.begin(), weights.end());
  
  uniform_real_distribution<double> random_weight(0.0, max_weight);
  uniform_int_distribution<int> particle_index(0, num_particles - 1);
  int idx = particle_index(gen);
  double beta = 0.0;
  
  vector<Particle> resampled_particles;
  unsigned int nParticles = particles.size();
  for (int i = 0; i < nParticles; i++) {
    beta += random_weight(gen) * 2.0;
    
    while (beta > weights[idx]) {
      beta -= weights[idx];
      idx = (idx + 1) % num_particles;
    }
    
    resampled_particles.push_back(particles[idx]);
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
  
  //Clear the previous associations
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