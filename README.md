Set 19
---

### 1. Create a `courses` array and render using `*ngFor`

**app.component.ts**
```ts
export class AppComponent {
  courses = ['Angular', 'React', 'Vue'];
}
```

**app.component.html**
```html
<ul>
  <li *ngFor="let course of courses">{{ course }}</li>
</ul>
```

---

### 2. Apply multiple CSS classes using `ngClass`

**styles.css**
```css
.highlight {
  color: red;
}
.bold {
  font-weight: bold;
}
```

**app.component.html**
```html
<p [ngClass]="['highlight', 'bold']">
  This text has multiple classes.
</p>
```

---

### 3. Display product code in lowercase and product name in uppercase

**app.component.ts**
```ts
export class AppComponent {
  productCode = 'XYZ123';
  productName = 'sample product';
}
```

**app.component.html**
```html
<p>Code: {{ productCode | lowercase }}</p>
<p>Name: {{ productName | uppercase }}</p>
```

---
