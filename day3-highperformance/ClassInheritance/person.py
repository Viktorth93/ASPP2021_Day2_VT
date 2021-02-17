
class Person(object):
    def __init__(self, first, last):
        ''' Constructor for class Person '''
        self.firstName = first
        self.lastName = last

    def printName(self):
        print(self.firstName + " " + self.lastName)

class Student(Person):
    def __init__(self, first, last, subject):
        #Person.__init__(self,first,last)
        super(Student, self).__init__(first, last)
        self.subject = subject

    def printNameSubject(self):
        print(self.firstName + " " + self.lastName + ", " + self.subject)

class Teacher(Person):
    def __init__(self, first, last, courseSubject):
        super(Teacher, self).__init__(first, last)
        self.courseSubject = courseSubject

    def printNameCourse(self):
        print(self.firstName + " " + self.lastName + ", " + self.courseSubject)
