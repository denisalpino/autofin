Быстрый старт
=============

Этот гайд поможет вам быстро установить и начать использовать проект.

Установка
---------

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install autofin

Основное использование
-------------------------------

.. code-block:: python

   from autofin.preprocessing.cv import GroupTimeSeriesSplit

   # Пример использования класса
   splitter = GroupTimeSeriesSplit(
       val_folds=2,
       interval="7d",
       window="expanding"
   )
   # ...