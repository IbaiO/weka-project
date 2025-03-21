import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.SerializationHelper;

public class FSSetaNB {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.out.println("Ez duzu sartu beharreko parametro guztiak");
            System.exit(1);
        }
        String trainArff = args[0];
        String NBmodelPath = args[1];
        String headersPath = args[2];

        DataSource source = new DataSource(trainArff); // "home/mikel/Desktop/data/5/data_supervised.arff"
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
        System.out.println( "\n");

        // Train a NaiveBayes model using the train
        NaiveBayes nb = new NaiveBayes();   

        // Apply the filter to the training set
        Instances filteredTrain = attributeSelection.reduceDimensionality(train);
        System.out.println("Number of attributes after feature selection: " + filteredTrain.numAttributes());

        // Save the filtered train set to ARFF file
        //BufferedWriter writer = new BufferedWriter(new FileWriter(headersPath));
        //writer.write(filteredTrain.toString());
        //writer.flush();
        //writer.close();
        //System.out.println("Filtered train set saved to: " + headersPath);

        // Save the headers of filtered train set to text file
        //BufferedWriter writer = new BufferedWriter(new FileWriter(headersPath));
        //for (int i = 0; i < filteredTrain.numAttributes(); i++) {
        //    writer.write(filteredTrain.attribute(i).name() + "\n");
        //}
        //writer.flush();
        //writer.close();
        //System.out.println("Headers of filtered train set saved to: " + headersPath);

        // Save the headers of filtered train set to text file
        BufferedWriter writer = new BufferedWriter(new FileWriter(headersPath));
        for (int i = 0; i < filteredTrain.numAttributes(); i++) {
            writer.write(filteredTrain.attribute(i).name() + "\n");
        }
        writer.flush();
        writer.close();
        System.out.println("Headers of filtered train set saved to: " + headersPath);
        
        // Build the classifier
        nb.buildClassifier(filteredTrain);
        System.out.println("NaiveBayes model trained");

        // Save the trained model
        SerializationHelper.write(NBmodelPath, nb);
        System.out.println("NaiveBayes model saved to: " + NBmodelPath);

    }
}
