package src;

import libsvm.svm;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class FSSetaSVM {

    public static void main(String[] args) {

        if (args.length != 1) {
            System.out.println("Sartu ezazu arff bat.");
            System.exit(1);
        }
        String data = args[0];
        DataSource source = new DataSource(data); 
        Instances train = source.getDataSet();

        // Create a new SVM
        svm mySVM = new svm();
        svm_train train2 = new svm_train();

        // Set the SVM options
        mySVM.svm_check_parameter(new String[]{"-S", "0", "-K", "2", "-D", "3"});
        train2.svm_train(train, mySVM);

        // Print the SVM options
        System.out.println(mySVM.svm_get_nr_class());

        // Train the SVM
        mySVM.svm_train(data);

        // Print the SVM
        System.out.println(mySVM);
    }
}