import matplotlib.pyplot as plt
import scikitplot as skplt

# This is a Keras classifier. We'll generate probabilities on the test set.
keras_clf.fit(X_train, y_train, batch_size=64, nb_epoch=10, verbose=2)
probas = keras_clf.predict_proba(X_test, batch_size=64)
# Now plot.
skplt.metrics.plot_precision_recall_curve(y_test, probas)
plt.show()
