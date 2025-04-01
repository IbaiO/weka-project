package src;

import weka.classifiers.functions.SMO;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class sMO {

    public static SMO main(Instances data) throws Exception {

        // Load the training data
        Instances train = data;
        if (train == null) {
            System.out.println("Error: Unable to load training data.");
            return null;
        }
        
        // Check if there are string attributes
        boolean hasStringAttributes = false;
        for (int i = 0; i < train.numAttributes(); i++) {
            if (train.attribute(i).isString()) {
                    hasStringAttributes = true;
                    break;
                }
            }

            if (hasStringAttributes) {
                // Apply StringToWordVector filter
                StringToWordVector stringToWordVector = new StringToWordVector();
                stringToWordVector.setInputFormat(train);
                train = Filter.useFilter(train, stringToWordVector);
            }

        // Set the class index to the last attribute
        if (train.classIndex() == -1) {
            train.setClassIndex(train.numAttributes() - 1);
        }

        System.out.println("Total number of attributes: " + train.numAttributes());

        // Find the best parameters for the SMO classifier using CVParameterSelection
        CVParameterSelection cvParamSelection = new CVParameterSelection();
        cvParamSelection.setClassifier(new SMO());
        
        // Select the parameters to optimize
        cvParamSelection.addCVParameter("C 0.1 2.0 10"); // Probar valores de C entre 0.1 y 2.0 en 10 pasos
        cvParamSelection.addCVParameter("K \"weka.classifiers.functions.supportVector.PolyKernel -E 1\" \"weka.classifiers.functions.supportVector.RBFKernel -G 0.01\""); // Probar diferentes kernels
        cvParamSelection.addCVParameter("E 1 3 3"); // Probar grados del kernel polinómico entre 1 y 3
        cvParamSelection.addCVParameter("G 0.01 1.0 5"); // Probar valores de gamma entre 0.01 y 1.0 en 5 pasos
        cvParamSelection.addCVParameter("L 0.001 0.1 5"); // Probar tolerancia entre 0.001 y 0.1 en 5 pasos

        // Search for the best parameters
        cvParamSelection.buildClassifier(train);

        // Lortu aurkitutako parametro onenak
        String bestOptions = cvParamSelection.toString();
        System.out.println("Best parameters found: " + bestOptions);

        // SMO modeloa sortu aurkitutako parametro onenekin
        SMO bestSMO = new SMO();
        bestSMO.setOptions(weka.core.Utils.splitOptions(bestOptions));

        // Train the SMO model with the training data
        bestSMO.buildClassifier(train);

        // Modeloa itzuli
        return bestSMO;
    }
}

