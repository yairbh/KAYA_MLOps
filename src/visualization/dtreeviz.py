#  !pip install -q dtreeviz   # add to requirements

%config InlineBackend.figure_format = 'retina' # Make visualizations look good
#%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import pandas as pd
import dtreeviz


def dtreeviz_analysis(X_train_clean, y_train_clean, target_column_name, model, tree_index = 1,  class_names=[0, 1],
                      entire_tree = True, instance_to_query = None, nodes_to_query = None ):
  '''
  this function recives the training data and target, a trained model and a class name (as a list ['no', 'yes'], default is [0,1]).
  it returms a dtreeviz visualisation for the model.
  tree_index specifies which individual tree within the ensemble to visualize (tree_index = 1 is default).
  if entire_tree = True, it will return the entire tree (True is default).
  if nodes_to_query is a list of two int, it will also return this specific section of the tree (root is level 0).
  if instance_to_query is an int, it will also return the specific path of that instance.
  '''

  viz_model = dtreeviz.model(model, X_train=X_train_clean, y_train=y_train_clean,
                           feature_names=list(X_train_clean.columns), tree_index=tree_index,
                           target_name=target_column_name, class_names=[class_names[0], class_names[1]])



  # Display visualisation for the entire tree
  if entire_tree:
    print("Displaying entire tree:")
    display(viz_model.view(orientation="LR", fontname='DejaVu Sans'))

  # display visualisation for a section of the tree (need to specify which nodes)
  if nodes_to_query is not None:
    nodes = nodes_to_query
    print(f"Displaying levels {nodes[0]} to {nodes[1]}:")
    display(viz_model.view(depth_range_to_display=(nodes[0], nodes[1]), orientation="LR", fontname='DejaVu Sans'))


  # Display visualisation fo for a specific instance
  if instance_to_query is not None:
    i = instance_to_query
    x = X_train_clean.iloc[i]
    print(f"path for instance {instance_to_query}:")
    display(viz_model.view(x=x, orientation="LR", fontname='DejaVu Sans'))

    # # Print the specific path
    # viz_model.view(x=x, show_just_path=True, orientation="LR")
    # display(viz_model.explain_prediction_path(x))  # print it as list of features and values

  return