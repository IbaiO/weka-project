import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

public class iragarpenakEgin {
    
    public static void main(String[] args) throws Exception {
        if(args.length < 3) {
            System.out.println("Ez duzu sartu beharreko parametro guztiak");
            System.exit(1);
        }
        String blindTestArff = args[0];
        String NBmodelPath = args[1];
        String trainHeadersPath = args[2];

        // Load the test set
        DataSource source = new DataSource(blindTestArff);
        Instances test_set = source.getDataSet();

        // Load the headers from the train_headers.txt
        Set<String> headersSet = new HashSet<>();
        try (BufferedReader br = new BufferedReader(new FileReader(trainHeadersPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                headersSet.add(line.trim());
            }
        }

        // Ensure the test set has the same structure as the headers set
        System.out.println("Number of attributes in the test set before removing extra attributes: " + test_set.numAttributes());
        System.out.println("Headers set: " + headersSet);
        StringBuilder indicesToRemove = new StringBuilder();
        for (int i = 0; i < test_set.numAttributes(); i++) {
            if (!headersSet.contains(test_set.attribute(i).name().trim())) {
                indicesToRemove.append(i + 1).append(",");
            }
        }
        if (indicesToRemove.length() > 0) {
            indicesToRemove.setLength(indicesToRemove.length() - 1); // Remove the last comma
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndices(indicesToRemove.toString());
            removeFilter.setInputFormat(test_set);
            test_set = Filter.useFilter(test_set, removeFilter);
        }

        System.out.println("Number of attributes after removing extra attributes: " + test_set.numAttributes());

        // Set the class index to the last attribute
        if (test_set.classIndex() == -1) {
            test_set.setClassIndex(test_set.numAttributes() - 1);
        }

        // Load the Naive Bayes model
        Classifier modeloa = (Classifier) SerializationHelper.read(NBmodelPath);

        // Make predictions
        for (int i = 0; i < test_set.numInstances(); i++) {
            double label = modeloa.classifyInstance(test_set.instance(i));
            test_set.instance(i).setClassValue(label);
            try (BufferedWriter writer = new BufferedWriter(new FileWriter("/home/mikel/Desktop/KIWI-WEKA/ARIKETA5/predictions.txt", true))) {
                writer.write("Instance " + (i + 1) + ": Predicted label = " + test_set.classAttribute().value((int) label));
                writer.newLine();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
