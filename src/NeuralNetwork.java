import java.util.Arrays;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;
    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;
    boolean first = true;

    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights, double[][] initial_output_layer_weights, double learning_rate) {
        //Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;

        this.learning_rate = learning_rate;
    }


    //Calculate neuron activation for an input
    public double sigmoid(double input) {
        double output = Double.NaN; //TODO!

        output = 1 / (1+Math.exp(-input));
        return output;
    }

    //Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {

        double[] hidden_layer_outputs = new double[num_hidden];

        for (int i = 0; i < num_hidden; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            double output = 0;

            for (int j = 0; j < inputs.length; j++) {
                weighted_sum += (hidden_layer_weights[j][i] * inputs[j]);

            }

            output = sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;

        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            double output = 0;

            for(int j = 0; j < hidden_layer_outputs.length; j++){
                weighted_sum += (output_layer_weights[j][i] * hidden_layer_outputs[j]);
            }

            output = sigmoid(weighted_sum);
            output_layer_outputs[i] = output;
        }

        return new double[][]{hidden_layer_outputs, output_layer_outputs};
    }

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs,
                                                 double[] output_layer_outputs, int[] desired_outputs) {


            double[] output_layer_betas = new double[num_outputs];
            // TODO! Calculate output layer betas.

            double outputBeta;
            for (int i = 0; i < num_outputs; i++) {
                outputBeta = desired_outputs[i] - output_layer_outputs[i];
                output_layer_betas[i] = outputBeta;
            }

           // System.out.println("OL betas: " + Arrays.toString(output_layer_betas));

            double[] hidden_layer_betas = new double[num_hidden];
            // TODO! Calculate hidden layer betas.

            double hiddenBeta;
            for(int i = 0; i < num_hidden; i++) {
                hiddenBeta = 0;
                for (int j = 0; j < num_outputs; j++) {
                    hiddenBeta += (output_layer_weights[i][j] * output_layer_outputs[j] *
                            (1 - output_layer_outputs[j]) * output_layer_betas[j]);
                    }
                hidden_layer_betas[i] = hiddenBeta;
            }

           //System.out.println("HL betas: " + Arrays.toString(hidden_layer_betas));

            // This is a HxO array (H hidden nodes, O outputs)
            double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];
            // TODO! Calculate output layer weight changes.

            for(int i = 0; i < num_hidden; i++){
                for(int j = 0; j < num_outputs; j++){
                    delta_output_layer_weights [i][j] = learning_rate * hidden_layer_outputs[i] *
                            output_layer_outputs[j] * (1 - output_layer_outputs[j]) * output_layer_betas[j];
                }
            }

            // This is a IxH array (I inputs, H hidden nodes)h
            double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
            // TODO! Calculate hidden layer weight changes.

            for(int i = 0; i < num_inputs; i++){
                for(int j = 0; j < num_hidden; j++){
                    delta_hidden_layer_weights [i][j] = (learning_rate * inputs[i] *
                            hidden_layer_outputs[j] * (1 - hidden_layer_outputs[j]) * hidden_layer_betas[j]);
                }
            }

        // Return the weights we calculated, so they can be used to update all the weights.
        return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights};
    }

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights) {
        // TODO! Update the weights

        for(int i = 0; i < num_hidden; i++){
            for(int j = 0; j < num_outputs; j++){
                this.output_layer_weights[i][j] += delta_output_layer_weights[i][j];
            }

        }
        for(int i = 0; i < num_inputs; i++){
            for(int j = 0; j < num_hidden; j++){
                this.hidden_layer_weights[i][j] += delta_hidden_layer_weights[i][j];
            }
        }

    }

    public void train(double[][] instances, int[][] desired_outputs, int epochs) {

        for (int epoch = 0; epoch < epochs; epoch++) {
            double correct = 0;
            System.out.println("epoch = " + epoch);
            int[] predictions = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i]);

                int predicted_class = -1; // TODO! done

                predicted_class = predict(instances)[i];
                predictions[i] = predicted_class;

                //We use online learning, i.e. update the weights after every instance.
                update_weights(delta_weights[0], delta_weights[1]);
            }

            // Print new weights

            System.out.println("Hidden layer weights \n" + Arrays.deepToString(hidden_layer_weights));
            System.out.println("Output layer weights  \n" + Arrays.deepToString(output_layer_weights));

            // TODO: Print accuracy achieved over this epoch
            double acc = Double.NaN;

            for(int i = 0; i<instances.length; i++){
                if(predictions[i] == 0 && desired_outputs[i][0] == 1) {correct++;}
                else if(predictions[i] == 1 && desired_outputs[i][1] == 1) {correct++;}
                else if(predictions[i] == 2 && desired_outputs[i][2] == 1) {correct++;}
            }

            acc = (correct/instances.length) * 100;
            System.out.println("acc = " + acc);
        }
    }

    public int[] predict(double[][] instances) {

        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            if(first) {
                System.out.println("\noutputs for first instance:");
                System.out.println("Hidden Layer outputs: " + outputs[0][0] +" " + outputs[0][1]);
                System.out.println("Output Layer outputs: " + outputs[1][0] +" " + outputs[1][1] + " " + outputs[1][2] + "\n");

            }
            first=false;
            int predicted_class = -1;  // TODO !Should be 0, 1, or 2.
            double max = Double.MIN_VALUE;

            for(int j = 0; j < outputs[outputs.length-1].length; j++){ //finds max output
                if(outputs[outputs.length-1][j] > max) {
                    max = outputs[outputs.length - 1][j];
                    predicted_class = j;
                }
            }
            predictions[i] = predicted_class;
        }
        return predictions;
    }

}

