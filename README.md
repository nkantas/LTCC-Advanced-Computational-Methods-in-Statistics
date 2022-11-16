# Advanced Computational Methods in Statistics

This course will provide an overview of Monte Carlo methods when used for problems in Statistics. After an introduction to simulation, its purpose and challenges, we will cover in more detail Importance Sampling, Markov Chain Monte Carlo and Sequential Monte Carlo. Whilst the main focus will be on the methodology and its relevance to applications, we will often mention relevant theoretical results and their importance for problems in practice. 

Keywords: Simulation, Variance reduction, Monte Carlo methods, Importance Sampling, Markov Chain Monte Carlo, Sequential Monte Carlo, Particle filters, Stochastic filtering, Hidden Markov models.

Part of the material below is delivered every year as part of the London Taught Course Centre [(LTCC)](http://www.ltcc.ac.uk/) for PhD students in mathematical sciences. Here is the official link of the course: http://www.ltcc.ac.uk/courses/advanced-computational-methods-in-statistics/

The lecture notes can be found [here](advanced_simulation_notes_ltcc.pdf) and the Matlab code used in the examples is provided above. 

Immediately below you can find the course outline together the with links to slides and video-lectures (~20hrs). I have listed all the video lectures in this [playlist](https://www.youtube.com/playlist?list=PLnLW5bw8rfk3Tt4K8YrH7tPyLH9XKzGo2). At the bottom of this page you can find more details on assessment, reading list, prerequisites, delivery format for the LTCC course.

# Introduction

We will split the introduction into two parts:

- Introduction to Monte Carlo  [slides](slides/intro_mc.pdf) and [video](https://www.youtube.com/watch?v=uqDItPClDiM)

- Basic Indirect Sampling methods (Rejection Sampling and Importance Sampling)  [slides](slides/intro_mc2.pdf) and [video](https://www.youtube.com/watch?v=m5Nt3GQFj3Y)

# Markov Chain Monte Carlo (MCMC)

We will mainly discuss various topics and provide some some basics on theory and practice.

- Introduction to Markov Chain Monte Carlo 
  - Metropolis-Hastings, Gibbs sampling, diagnostics 
  - [video](https://youtu.be/gCbzRfAA70g) and [slides](slides/mcmc_intro.pdf)   
- Some more methodology 
  - Computing the normalising constant, Adaptive MCMC, Pseudo marginal MCMC
  - [video](https://youtu.be/M_R-IiiSF4Q) and [slides](slides/mcmc_extensions.pdf)
- Theoretical topics
  - Understanding MCMC from basics of Markov Chain theory
  - Diffusions, MCMC and the 0.234 rule of thumb
  - [video](https://youtu.be/5pF8VmJqtak)  and [slides](slides/mcmc_theory.pdf) and [scribles](slides/characterisation_of_pi.pdf)
   
# Hidden Markov models and the filtering problem

- Hidden Markov models and filtering: [slides](slides/HMMs_Filtering.pdf) and [video](https://youtu.be/GnlWK1erBmc)

- The Kalman filter and extensions: [slides](slides/Kalman.pdf) and [video](https://youtu.be/g6h3gCp2tcM)

# Sequential Monte Carlo (SMC)
  
  - Sequential Importance sampling: [slides](slides/intro_mc3.pdf) and [video](https://youtu.be/MU0QnWU9ULM)
  
  - Introduction to Particle filtering: [slides](slides/Intro_PF.pdf) and [video](https://youtu.be/Vkc3lqs1YQo)
  
  - Some extensions to the basic Particle filter 
     - adaptive resampling, resample move and auxiliary particle filters   
     - [slides](slides/more_advanced_pf.pdf) and [video](https://youtu.be/n4ouaf_K2KU)

  - Particle Smoothing: [slides](slides/Particle_Smoothing.pdf) and [video](https://youtu.be/-1XeSWNOuRk)
  
  - Parameter estimation methods for static parameters of Hidden Markov models
  
      - Bayesian methods and particle MCMC: [slides](slides/BayesianHMM.pdf) and [video](https://youtu.be/_Rl27OoCWKs)
      - likelihood estimation: [slides](slides/likelihoodHMM.pdf) and [video](https://youtu.be/sqbst6hyX6w)
      
  - SMC sampling for fixed dimensional state spaces: [slides](slides/smc_fixed_space.pdf) and [video](https://youtu.be/ocuKMctUndg)
     
  
# LTCC Course organisation and assessment

## Delivery

The course will be delivered in person at Hardy room in De Morgan House every Monday 10:50-12:50 from 7 November to 5 December 2022.

<!--
The lecture room is Room 140, Huxley Building, Imperial College London, South Kensington Campus

For the streaming please visit:
https://imperial-ac-uk.zoom.us/j/93135172628?pwd=cUR3aHpsZFZUblV3S0RXdUZZS2tCZz09
---Meeting ID: 931 3517 2628
Passcode: CmnD?4 

Lecture Recordings are available upon request (due to limited file size hosting capacity).

Note there will be significant overlap with linked videos above, and my recommendation is to use the pre-recorded ones linked above with the slides when a large part of the lecture is viewed online.

-->

Registration is compulsory, please vist http://www.ltcc.ac.uk/registration/

## Coursework

You may find the 2020 coursework instructions [here](http://wwwf.imperial.ac.uk/~nkantas/Coursework.pdf)

For the 2022 assessment more details will appear here soon. 

<!---Deadline: around 8 December (about a month)

Page limit: 10 pages, recommended length around 6-8 pages

Submit in MS Teams assingment, if there are any issues email to n.kantas at imperial.ac.uk _*using subject: LTCC coursework submission*_ -->
  
## References   
  
Relevant introductory graduate textbooks and edited volumes:

  - Chopin and Papaspiliopoulos (2020). An introduction to sequential Monte Carlo, Springer Series in Statistics.
  
  - R. Douc, E. Moulines, & D. S. Stoffer (2014). Nonlinear Time Series Theory, Methods and Applications with R Examples, CRC Press.
  
  - S. Sarkka (2013) Bayesian filtering and smoothing, CUP Cambridge.  
  
  - Cappé, O., Moulines, E. and Rydén, T. (2005). Inference in Hidden Markov Models. New York: Springer-Verlag. 
  
  - Doucet, de Freitas, Gordon (2001) Sequential Monte Carlo Methods in Practice, Springer.

  - Liu (2001) Monte Carlo strategies in scientific computing, Springer.
  
  - Robert and Casella (1999) Monte Carlo Statistical Methods, Springer. 
  
  - Gillks, Richardson, Spiegelhalter (1996) Markov Chain Monte Carlo in Practice, Chapman Hall.

<!---## Older lecture slides

For you convenience i am listing the slides used in previous years. You might notice this year's course has been broken down and changed a bit.

 - [Slides 1](http://wwwf.imperial.ac.uk/~nkantas/slides1.pdf)
 - [Slides 2](http://wwwf.imperial.ac.uk/~nkantas/slides2.pdf)
 - [Slides 3](http://wwwf.imperial.ac.uk/~nkantas/slides3.pdf)
 - [Slides 4](http://wwwf.imperial.ac.uk/~nkantas/slides4.pdf)-->

## Prerequisites: 

 - Basic knowledge of Statistics and Probability. 
  
 - Basic knowledge of programming in any language appropriate for scientific computing.
  
 - Familiarity and exposure to Markov Chains or stochastic processes will be useful.

## Format:

- There will be optional exercises or small courseworks posed as quizes or homeworks. There will be no separate problem sheets. Some problems will require the use of some programming.

- Lecture/computer session/tutorial/discussion hours split: 10/ 0 /0 /0 

## Lecturer contact details:

  * Nikolas Kantas, Imperial College
 
  * Email: n.kantas at imperial.ac.uk

  * website: http://wwwf.imperial.ac.uk/~nkantas/

