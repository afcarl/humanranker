from django import forms
from django.contrib.auth.models import User
from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from ranker.models import Project, Item
from multiupload.fields import MultiFileField

class UserCreateForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def save(self, commit=True):
        user = super(UserCreateForm, self).save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user

class ProjectForm(ModelForm):
    items = MultiFileField(max_num=1000, min_num=2,
                                 max_file_size=1024*1024*1024)
    #items.required = False
    #attachements = forms.ImageField(required=False)

    class Meta:
        model = Project
        fields = ['name', 'pairwise_prompt', 'individual_likert_prompt',
                  'likert_label_7', 'likert_label_6', 'likert_label_5',
                  'likert_label_4', 'likert_label_3', 'likert_label_2',
                  'likert_label_1']

    def save_images(self):
        for item in self.cleaned_data['items']:
            it = Item(name=item, image=item, project=self.instance)
            it.save()

class ProjectUpdateForm(ModelForm):
    items = MultiFileField(max_num=1000, min_num=0,
                                 max_file_size=1024*1024*1024,required=False)

    class Meta:
        model = Project
        fields = ['name', 'pairwise_prompt', 'individual_likert_prompt',
                  'likert_label_7', 'likert_label_6', 'likert_label_5',
                  'likert_label_4', 'likert_label_3', 'likert_label_2',
                  'likert_label_1']
        
    def save_images(self):
        for item in self.cleaned_data['items']:
            it = Item(name=item, image=item, project=self.instance)
            it.save()

