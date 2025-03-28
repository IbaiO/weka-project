package src;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class FSSetaSMO {

    public static SMO main(String[] args) throws Exception {

        String traindata = args[0];
        DataSource source = new DataSource(traindata); 
        Instances train = source.getDataSet();

        // Set the class index to the last attribute
        if (train.classIndex() == -1) {
            train.setClassIndex(train.numAttributes() - 1);
        }

        System.out.println("Total number of attributes: " + train.numAttributes());

        // Use the AttributeSelection class to perform feature selection
        AttributeSelection attributeSelection = new AttributeSelection();
        CfsSubsetEval evaluator = new CfsSubsetEval();
        BestFirst search = new BestFirst();

        attributeSelection.setEvaluator(evaluator);
        attributeSelection.setSearch(search);
        attributeSelection.SelectAttributes(train);

        // Get the selected attributes
        int[] selectedAttributes = attributeSelection.selectedAttributes();
        System.out.println("Selected attributes: ");
        for (int i = 0; i < selectedAttributes.length; i++) {
            System.out.print(selectedAttributes[i] + " ");
        }

        // Apply the filter to the training set
        System.out.println("Number of attributes before feature selection: " + train.numAttributes());
        Instances filteredTrain = attributeSelection.reduceDimensionality(train);
        System.out.println("Number of attributes after feature selection: " + filteredTrain.numAttributes());

        // Train a SMO model using the train
        SMO smo = new SMO();
        smo.setC(1.0);
        smo.setKernel(new weka.classifiers.functions.supportVector.PolyKernel()); // other options: RBFKernel, LinearKernel
        smo.setEpsilon(0.001);
        smo.setToleranceParameter(0.001);

        smo.buildClassifier(train);
 
        // Build the model
        smo.buildClassifier(filteredTrain);
        System.out.println("Model built successfully.");
        System.out.println( "\n");

        // Return the trained model
        return smo;
    }
}

