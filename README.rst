|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Ускоренные методы нулевого порядка в гладкой выпуклой стохастической оптимизации
    :Тип научной работы: M1P
    :Автор: Фанис Адикович Хафизов
    :Научный руководитель: к.ф.-м.н. Безносиков Александр Николаевич
    :Научный консультант(при наличии): Богданов Александр Иванович

Abstract
========

Данная работа посвящена задачи оптимизации без доступа к градиенту целевой функции $f(x)$, из-за чего нам надо как-то его оценивать. Рассматривается безградиентный метод JAGUAR, использующий информацию о предыдущих вызовах и требует $\mathcal{O}(1)$ оракульных вызовов. Мы применяем эту аппроксимацию в ускоренном методе градиентного спуска и докажем его сходимость для выпуклой задачи оптимизации. Также сравним метод JAGUAR с другими известными способами на экспериментах.
